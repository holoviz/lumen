from __future__ import annotations

import asyncio
import hashlib

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import param

from panel_material_ui import Card, Column, Typography

from ..config import PROMPTS_DIR
from ..context import TContext
from ..llm import Message
from ..models import CallSpec, StepDecision, SufficiencyCheck
from ..report import ActorTask
from ..utils import log_debug, mutate_user_message, normalized_name
from .base import Coordinator, Plan

if TYPE_CHECKING:
    from ..agents import Agent


class HistoryEntry:
    """Tracks an actor's result at a specific detail level."""
    def __init__(self, summary: str, level: str, outputs: list, out_ctx: dict, task: ActorTask | None = None):
        self.summary = summary
        self.level = level
        self.outputs = outputs
        self.out_ctx = out_ctx
        self.task = task  # Track the ActorTask for building the final Plan


class IterativePlanner(Coordinator):
    """
    Iterative planner that executes one step at a time with auto-escalation.
    
    Unlike the standard Planner which generates a full plan upfront,
    IterativePlanner uses a loop:
    1. Decide next action (call/escalate/done)
    2. Execute action
    3. Router checks if result is sufficient
    4. Update history
    5. Repeat until done
    """

    max_iterations = param.Integer(default=15, doc="""
        Maximum number of iterations before forcing completion.""")

    max_calls_per_actor = param.Integer(default=3, doc="""
        Maximum times a single actor can be called.""")

    max_history_tokens = param.Integer(default=4000, doc="""
        Approximate token budget for history in prompts.""")

    planner_model = param.String(default=None, doc="""
        Model to use for planning decisions. If None, uses default LLM.""")

    prompts = param.Dict(default={
        "step": {
            "template": PROMPTS_DIR / "IterativePlanner" / "step.jinja2",
            "response_model": StepDecision,
        },
        "sufficiency": {
            "template": PROMPTS_DIR / "IterativePlanner" / "sufficiency.jinja2",
            "response_model": SufficiencyCheck,
        },
        # Add empty main to satisfy base Coordinator's _process_prompts
        "main": {},
    })

    def __init__(self, **params):
        super().__init__(**params)
        self._reset_state()

    def _reset_state(self):
        """Reset planner state for a new query."""
        self._actor_history: dict[str, HistoryEntry] = {}
        self.actor_call_counts: dict[str, int] = defaultdict(int)
        self.call_hashes: set[str] = set()
        self.warnings: list[str] = []
        self.all_outputs: list[Any] = []
        self._iteration_history: list[str] = []  # Track completed iterations for UI
        self._executed_tasks: list[ActorTask] = []  # Track tasks for final Plan

    def _hash_call(self, actor: str, instruction: str) -> str:
        """Create hash for deduplication."""
        content = f"{actor}:{instruction[:100].lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _truncate_history(self, history: dict[str, HistoryEntry], prefer_level: str = "bare") -> dict[str, dict]:
        """
        Truncate history to fit token budget.
        """
        total_chars = 0
        max_chars = self.max_history_tokens * 4  # ~4 chars per token

        truncated = {}
        for actor in reversed(list(history.keys())):
            entry = history[actor]

            actor_obj = self._actors_dict.get(actor)
            if actor_obj:
                all_summaries = actor_obj.summarize(entry.outputs, entry.out_ctx)
            else:
                all_summaries = {"bare": entry.summary, "compact": entry.summary, "detailed": entry.summary}

            summary_to_show = all_summaries.get(prefer_level, entry.summary)
            summary_len = len(summary_to_show)

            if total_chars + summary_len > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    truncated[actor] = {
                        "summary": summary_to_show[:remaining] + "...",
                        "level": prefer_level,
                    }
                break

            truncated[actor] = {
                "summary": summary_to_show,
                "level": prefer_level,
                "stored_level": entry.level,
            }
            total_chars += summary_len

        return dict(reversed(list(truncated.items())))

    def _render_iteration_status(self, current_iteration: int, current_action: str = "", status: str = "running") -> str:
        """Render the current iteration status for the UI."""
        lines = []
        for entry in self._iteration_history:
            lines.append(f"- [x] {entry}")

        if status == "running" and current_action:
            lines.append(f"- [ ] **{current_action}**")
        elif status == "success" and current_action:
            lines.append(f"- [x] {current_action}")

        return "\n".join(lines) if lines else "Starting..."

    async def _check_sufficiency(
        self,
        instruction: str,
        result: str,
        original_query: str
    ) -> bool:
        """Router: check if result is sufficient to proceed."""
        response = await self._invoke_prompt(
            "sufficiency",
            messages=[],
            context={},
            instruction=instruction,
            result=result,
            original_query=original_query,
        )
        return response.sufficient

    async def _call_with_router(
        self,
        call: CallSpec,
        messages: list[Message],
        context: TContext,
        original_query: str,
    ) -> tuple[list[Any], dict, str, str, ActorTask] | None:
        """
        Call an agent and auto-escalate based on router decision.
        
        Returns tuple of (outputs, out_ctx, summary, level, task) or None if blocked
        """
        actor_name = call.actor

        # Check call limits
        if self.actor_call_counts[actor_name] >= self.max_calls_per_actor:
            self.warnings.append(
                f"{actor_name} already called {self.max_calls_per_actor}× — escalate existing result or move on"
            )
            return None

        # Check for duplicate calls
        call_hash = self._hash_call(actor_name, call.instruction)
        if call_hash in self.call_hashes:
            self.warnings.append(
                f"Similar {actor_name} call already made — use existing result or try different approach"
            )
            return None

        # Get agent or tool
        actor = self._actors_dict.get(actor_name)
        if not actor:
            self.warnings.append(f"Unknown actor: {actor_name}")
            return None

        # Create ActorTask for proper UI streaming and Plan integration
        task = ActorTask(
            actor,
            instruction=call.instruction,
            title=f"{actor_name}: {call.instruction[:50]}{'...' if len(call.instruction) > 50 else ''}",
        )

        # Prepare messages with instruction
        agent_messages = mutate_user_message(
            f"\n\nInstruction: {call.instruction}",
            messages.copy()
        )

        try:
            with task.param.update(
                interface=self.interface,
                llm=actor.llm or self.llm,
                steps_layout=self.steps_layout,
                history=messages,
            ):
                outputs, out_ctx = await task.execute(context)
        except Exception as e:
            out_ctx = {"__error__": str(e)}
            outputs = []
            task.status = "error"
            log_debug(f"[{actor_name}] Error: {e}")

        # Track call
        self.actor_call_counts[actor_name] += 1
        self.call_hashes.add(call_hash)

        # Get summaries
        summaries = actor.summarize(outputs, out_ctx)

        # Start with compact (skip bare router check to save tokens)
        level = "compact"
        if "__error__" not in out_ctx:
            if not await self._check_sufficiency(call.instruction, summaries["compact"], original_query):
                level = "detailed"

        return outputs, out_ctx, summaries[level], level, task

    def _escalate(self, actor_name: str) -> tuple[str, str] | None:
        """Escalate to next detail level for an actor."""
        if actor_name not in self._actor_history:
            self.warnings.append(f"Cannot escalate {actor_name} — no previous result")
            return None

        entry = self._actor_history[actor_name]
        current_level = entry.level

        if current_level == "detailed":
            self.warnings.append(f"{actor_name} already at detailed level — use data or move on")
            return None

        next_level = {"bare": "compact", "compact": "detailed"}[current_level]

        agent = self._agents_dict.get(actor_name)
        if not agent:
            return None

        summaries = agent.summarize(entry.outputs, entry.out_ctx)
        new_summary = summaries[next_level]

        self._actor_history[actor_name] = HistoryEntry(
            summary=new_summary,
            level=next_level,
            outputs=entry.outputs,
            out_ctx=entry.out_ctx,
            task=entry.task,
        )

        return new_summary, next_level

    async def _plan_step(
        self,
        messages: list[Message],
        context: TContext,
        original_query: str,
    ) -> StepDecision:
        """Get next step decision from LLM."""
        truncated_history = self._truncate_history(self._actor_history)

        agents_info = {name: agent for name, agent in self._agents_dict.items()}
        tools_info = {name: tool for name, tool in self._tools_dict.items()}

        response = await self._invoke_prompt(
            "step",
            messages=[{"role": "user", "content": "What's the next step?"}],
            context=context,
            original_query=original_query,
            agents=agents_info,
            tools=tools_info,
            history=truncated_history,
            warnings=self.warnings,
            model_spec=self.planner_model,
        )

        self.warnings = []
        return response

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        **kwargs
    ) -> Plan | None:
        """
        Main loop: iteratively plan and execute until done.
        Returns a Plan with pre-executed tasks for UI integration.
        """
        self._reset_state()

        # Normalize messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            messages = [messages]

        # Extract original query
        original_query = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                original_query = msg.get("content", "")
                break
            elif isinstance(msg, str):
                original_query = msg
                break

        log_debug(f"IterativePlanner starting: {original_query[:100]}", show_sep="above")

        # Convert agents/tools to dicts
        self._agents_dict = {normalized_name(agent): agent for agent in self.agents}
        self._tools_dict = {}
        for tool_list in self._tools.values():
            for tool in tool_list:
                self._tools_dict[normalized_name(tool)] = tool
        self._actors_dict = {**self._agents_dict, **self._tools_dict}

        # Set up UI steps layout
        self._todos_title = Typography(
            "🔄 Iterative planning...",
            css_classes=["todos-title"],
            margin=0,
            styles={"font-weight": "normal", "font-size": "1.1em"}
        )
        todos = Typography(
            css_classes=["todos"],
            margin=0,
            styles={"font-weight": "normal"}
        )
        todos_layout = Column(
            self._todos_title,
            todos,
            stylesheets=[".markdown { padding-inline: unset; }"]
        )
        self.steps_layout = Card(
            header=todos_layout,
            collapsed=not self.verbose,
            elevation=2
        )
        self.interface.stream(self.steps_layout, user="Planner")

        final_status = "success"

        for iteration in range(self.max_iterations):
            log_debug(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Update UI
            self._todos_title.object = f"🔄 Iteration {iteration + 1}/{self.max_iterations}"
            todos.object = self._render_iteration_status(iteration, "Deciding next step...")

            # Get next step decision
            with self._add_step(title="Deciding next action...", user="Planner", steps_layout=self.steps_layout) as step:
                step_decision = await self._plan_step(messages, context, original_query)
                step.stream(f"**Thinking:** {step_decision.thinking[:200]}...")
                step.success_title = f"Decision: {step_decision.action}"

            log_debug(f"StepDecision: action={step_decision.action}, actor={step_decision.actor}")

            # Handle done
            if step_decision.action == "done":
                self._iteration_history.append("✅ Done")
                log_debug("Planner done", show_sep="below")
                break

            # Handle escalate
            if step_decision.action == "escalate" and step_decision.actor:
                with self._add_step(title=f"Escalating {step_decision.actor}...", user="Planner", steps_layout=self.steps_layout) as step:
                    result = self._escalate(step_decision.actor)
                    if result:
                        summary, level = result
                        step.stream(f"Escalated to {level} level")
                        step.success_title = f"Escalated {step_decision.actor} → {level}"
                        self._iteration_history.append(f"📈 Escalated {step_decision.actor} to {level}")
                        log_debug(f"Escalated {step_decision.actor} to {level}")
                    else:
                        step.stream("Escalation not possible")
                        step.failed_title = f"Could not escalate {step_decision.actor}"
                continue

            # Handle call
            if step_decision.action == "call" and step_decision.calls:
                call_names = ", ".join(c.actor for c in step_decision.calls)
                todos.object = self._render_iteration_status(iteration, f"Calling: {call_names}")

                # Run calls in parallel
                tasks = [
                    self._call_with_router(call, messages, context, original_query)
                    for call in step_decision.calls
                ]
                results = await asyncio.gather(*tasks)

                any_success = False

                for call, result in zip(step_decision.calls, results, strict=False):
                    if result is None:
                        continue

                    any_success = True
                    outputs, out_ctx, summary, level, task = result

                    # Update history with task reference
                    self._actor_history[call.actor] = HistoryEntry(
                        summary=summary,
                        level=level,
                        outputs=outputs,
                        out_ctx=out_ctx,
                        task=task,
                    )

                    # Track executed task for final Plan
                    self._executed_tasks.append(task)

                    # Collect outputs
                    self.all_outputs.extend(outputs)

                    # Merge context
                    context.update(out_ctx)

                    # Add to iteration history
                    short_summary = summary[:60] + "..." if len(summary) > 60 else summary
                    self._iteration_history.append(f"📊 {call.actor}: {short_summary}")

                    log_debug(f"[{call.actor}] {level}: {summary[:100]}...")

                # Auto-finish for data-producing agents
                if any_success:
                    for call, result in zip(step_decision.calls, results, strict=False):
                        if result is None:
                            continue
                        outputs, out_ctx, summary, level, task = result

                        is_data_agent = call.actor in ("SQLAgent", "DbtslAgent", "AnalysisAgent")
                        has_data = "data" in out_ctx or "table" in out_ctx or "pipeline" in out_ctx
                        no_error = "__error__" not in out_ctx

                        if level == "compact" and is_data_agent and has_data and no_error:
                            log_debug(f"Auto-finishing: {call.actor} returned sufficient data")
                            self._iteration_history.append(f"✅ Auto-complete: sufficient data from {call.actor}")
                            break
                    else:
                        continue
                    break
        else:
            final_status = "warning"
            self.warnings.append("Max iterations reached")
            self._iteration_history.append("⚠️ Max iterations reached")
            log_debug("Max iterations reached", show_sep="below")

        # Update final UI status
        todos.object = self._render_iteration_status(iteration + 1, status="success")
        if final_status == "success":
            self._todos_title.object = f"✅ Completed in {len(self._iteration_history)} steps"
        else:
            self._todos_title.object = f"⚠️ Stopped after {iteration + 1} iterations"

        self.steps_layout.collapsed = True

        # Build and return a Plan with pre-executed tasks
        if self._executed_tasks:
            # Determine plan title from the query
            plan_title = original_query[:30] + "..." if len(original_query) > 30 else original_query

            plan = Plan(
                *self._executed_tasks,
                title=plan_title,
                history=messages,
                context=context,
                coordinator=self,
                steps_layout=self.steps_layout,
            )

            # Mark all tasks as pre-executed so Plan._run_task returns immediately
            for task in self._executed_tasks:
                if task.status != "error":
                    task.status = "success"

            return plan

        return None
