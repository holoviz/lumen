from __future__ import annotations

import asyncio
import traceback

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import param

from panel.io.state import state
from panel_material_ui import ChatStep
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from ..agents import Agent
from ..config import PROMPTS_DIR
from ..context import (
    LWW, ContextError, TContext, merge_contexts,
)
from ..llm import Message
from ..models import ThinkingYesNo
from ..report import ActorTask
from ..tools import MetadataLookup, Tool
from ..utils import log_debug, wrap_logfire
from .base import Coordinator, Plan

if TYPE_CHECKING:
    from panel.chat.step import ChatStep



class RawStep(BaseModel):
    actor: str
    instruction: str = Field(description="""
    Concise instruction capturing user intent at the right altitude.

    Right altitude:
    - ❌ Too low: implementation details (SQL syntax, chart specs, row limits, join on these tables)
    - ❌ Too high: vague ("handle this", "process data")
    - ✅ Just right: clear intent + relationship to context

    Encode intent: add/modify/filter/compare/create — reference existing context when applicable.

    Never include implementation details unless user explicitly specified them because selected actors
    will have more knowledge revealed during execution.
    """)
    title: str


class RawPlan(BaseModel):
    title: str = Field(description="A title that describes this plan, up to three words.")
    steps: list[RawStep] = Field(
        description="""
        A list of steps to perform that will solve user query. Each step MUST use a DIFFERENT actor than the previous step.
        Review your plan to ensure this constraint is met.
        """
    )


class Reasoning(BaseModel):
    chain_of_thought: str = Field(
        description="""
        Briefly summarize the user's goal and categorize the question type:
        high-level, data-focused, or other. Identify the most relevant and compatible actors,
        explaining their requirements, and what you already have satisfied. If there were previous failures, discuss them.
        IMPORTANT: Ensure no consecutive steps use the same actor in your planned sequence.
        """
    )


def make_plan_model(agents: list[str], tools: list[str]) -> type[RawPlan]:
    # TODO: make this inherit from PartialBaseModel
    step = create_model(
        "Step",
        actor=(Literal[tuple(agents+tools)], FieldInfo(description="The name of the actor to assign a task to.")),
        instruction=(str, FieldInfo(description="Instructions to the actor to assist in the task, and whether rendering is required.")),
        title=(str, FieldInfo(description="Short title of the task to be performed; up to three words.")),
    )
    plan = create_model(
        "Plan",
        title=(
            str, FieldInfo(description="A title that describes this plan, up to three words.")
        ),
        steps=(
            list[step],
            FieldInfo(
                description="""
                A list of steps to perform that will solve user query. Each step MUST use a DIFFERENT actor than the previous step.
                Review your plan to ensure this constraint is met.
                """
            )
        )
    )
    return plan


class Planner(Coordinator):
    """
    The Planner develops a plan to solve the user query step-by-step
    and then executes it.
    """

    planner_tools = param.List(
        default=[MetadataLookup],
        doc="""
        List of tools to use to provide context for the planner prior
        to making a plan.""",
    )

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "Planner" / "main.jinja2",
                "response_model": make_plan_model,
            },
            "follow_up": {
                "template": PROMPTS_DIR / "Planner" / "follow_up.jinja2",
                "response_model": ThinkingYesNo,
            },
        }
    )

    def __init__(self, **params):
        # Extract planner_tools before super().__init__ to prevent premature setting
        planner_tools_input = params.pop("planner_tools", self.param.planner_tools.default)

        # Initialize coordinator first so self.vector_store is available
        super().__init__(**params)

        # Now initialize planner_tools, reusing instances from _tools["main"] where possible
        if planner_tools_input:
            self.planner_tools = self._initialize_planner_tools(planner_tools_input, **params)

            # Prepare planner_tools (only those not already in _tools["main"])
            main_tool_ids = {id(tool) for tool in self._tools.get("main", [])}
            for tool in self.planner_tools:
                if id(tool) not in main_tool_ids:
                    state.execute(partial(tool.prepare, self.context))

    def _initialize_planner_tools(self, planner_tools_input: list, **params) -> list:
        """
        Initialize planner tools, reusing instances from _tools["main"] where possible.
        """
        # Build a mapping of tool types to instances from _tools["main"]
        main_tools_by_type = {}
        for tool in self._tools.get("main", []):
            tool_type = type(tool)
            if tool_type not in main_tools_by_type:
                main_tools_by_type[tool_type] = tool

        instantiated_tools = []
        for tool in planner_tools_input:
            if isinstance(tool, Tool):
                # Already instantiated - use as is
                instantiated_tools.append(tool)
            elif isinstance(tool, type):
                # Check if we already have an instance of this type in _tools["main"]
                if tool in main_tools_by_type:
                    # Reuse the existing instance
                    instantiated_tools.append(main_tools_by_type[tool])
                else:
                    # Initialize a new instance
                    init_params = {**params, "vector_store": self.vector_store}
                    tool_kwargs = self._get_tool_kwargs(tool, instantiated_tools, **init_params)
                    instantiated_tools.append(tool(**tool_kwargs))
            else:
                # Handle other cases (e.g., functions)
                init_params = {**params, "vector_store": self.vector_store}
                new_tools = self._initialize_tools_for_prompt([tool], **init_params)
                instantiated_tools.extend(new_tools)

        return instantiated_tools

    async def sync(self, context: TContext | None):
        """Sync both main tools and planner tools."""
        context = context or self.context
        synced_tools = set()

        # Sync main tools
        for tools in self._tools.values():
            for tool in tools:
                if id(tool) not in synced_tools:
                    await tool.sync(context)
                    synced_tools.add(id(tool))

        # Sync planner tools (skipping already synced ones)
        for tool in self.planner_tools:
            if id(tool) not in synced_tools:
                await tool.sync(context)
                synced_tools.add(id(tool))

    async def _check_follow_up_question(self, messages: list[Message], context: TContext) -> bool:
        """Check if the user's query is a follow-up question about the previous dataset."""
        # Only check if data is in memory
        if "data" not in context:
            return False

        # Use the follow_up prompt to check
        result = await self._invoke_prompt("follow_up", messages, context)

        is_follow_up = result.yes
        if not is_follow_up:
            context.pop("pipeline", None)

        return is_follow_up

    async def _execute_planner_tools(self, messages: list[Message], context: TContext):
        """Execute planner tools to gather context before planning."""
        if not self.planner_tools:
            return

        user_query = next((msg["content"] for msg in reversed(messages) if msg.get("role") == "user"), "")

        with self._add_step(title="Gathering context for planning...", steps_layout=self.steps_layout) as step:
            # Collect all output contexts for proper merging
            collected_contexts = []

            for tool in self.planner_tools:
                if not await tool.applies(context):
                    continue

                if tool.always_use:
                    is_relevant = True
                else:
                    is_relevant = await self._check_tool_relevance(tool, "", self, f"Gather context for planning to answer {user_query}", messages, context)

                if not is_relevant:
                    continue

                tool_name = getattr(tool, "name", type(tool).__name__)
                step.stream(f"Using {tool_name} to gather planning context...")
                task = ActorTask(
                    tool,
                    interface=self.interface,
                    instruction=user_query,
                    title=f"Gathering context with {tool_name}",
                    steps_layout=self.steps_layout,
                )
                outputs, out_context = await task.execute(context)
                if task.status == "error":
                    step.stream(f"\n\n✗ Failed to gather context from {tool_name}")
                    continue

                # Collect the output context
                collected_contexts.append(out_context)
                step.stream(f"\n\n✓ Collected {', '.join(f'`{k}`' for k in out_context)} from {tool_name}")

            # Merge all collected contexts using LWW (last-write-wins) strategy
            if collected_contexts:
                merged = merge_contexts(LWW, collected_contexts)
                context.update(merged)
                step.stream(f"\n\nMerged context keys: {', '.join(f'`{k}`' for k in merged)}")

    async def _pre_plan(
        self,
        messages: list[Message],
        context: TContext,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
    ) -> tuple[dict[str, Agent], dict[str, Tool], dict[str, Any]]:
        is_follow_up = await self._check_follow_up_question(messages, context)
        if not is_follow_up:
            await self._execute_planner_tools(messages, context)
        else:
            log_debug("\033[92mDetected follow-up question, using existing context\033[0m")
            with self._add_step(title="Using existing data context...", user="Assistant", steps_layout=self.steps_layout) as step:
                step.stream("Detected that this is a follow-up question related to the previous dataset.")
                step.stream("\n\nUsing the existing data in memory to answer without re-executing data retrieval.")
                step.success_title = "Using existing data for follow-up question"
        agents, tools, pre_plan_output = await super()._pre_plan(messages, context, agents, tools)
        pre_plan_output["is_follow_up"] = is_follow_up
        return agents, tools, pre_plan_output

    @wrap_logfire(span_name="Make Plan", extract_args=["messages", "previous_plans", "is_follow_up"])
    async def _make_plan(
        self,
        messages: list[Message],
        context: TContext,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
        unmet_dependencies: set[str],
        previous_actors: list[str],
        previous_plans: list[str],
        plan_model: type[RawPlan],
        step: ChatStep,
        is_follow_up: bool = False,
    ) -> BaseModel:
        agents = list(agents.values())
        tools = list(tools.values())
        all_provides = set()
        for provider in agents + tools:
            all_provides |= set(provider.output_schema.__annotations__)
        all_provides |= set(context)

        # filter agents using applies
        agents = [agent for agent in agents if await agent.applies(context)]

        # ensure these candidates are satisfiable
        # e.g. DbtslAgent is unsatisfiable if DbtslLookup was used in planning
        # but did not provide dbtsl_metaset
        # also filter out agents where excluded keys exist in context
        agents = [agent for agent in agents if len(set(agent.input_schema.__required_keys__) - all_provides) == 0 and type(agent).__name__ != "ValidationAgent"]
        tools = [tool for tool in tools if len(set(tool.input_schema.__required_keys__) - all_provides) == 0]
        reasoning = None
        while reasoning is None:
            # candidates = agents and tools that can provide
            # the unmet dependencies
            agent_candidates = [agent for agent in agents if not unmet_dependencies or set(agent.output_schema.__annotations__) & unmet_dependencies]
            tool_candidates = [tool for tool in tools if not unmet_dependencies or set(tool.output_schema.__annotations__) & unmet_dependencies]
            system = await self._render_prompt(
                "main",
                messages,
                context,
                agents=agents,
                tools=tools,
                unmet_dependencies=unmet_dependencies,
                candidates=agent_candidates + tool_candidates,
                previous_actors=previous_actors,
                previous_plans=previous_plans,
                is_follow_up=is_follow_up,
            )
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)

            async for reasoning in self.llm.stream(
                messages=messages,
                system=system,
                model_spec=model_spec,
                response_model=Reasoning,
                max_retries=3,
            ):
                if reasoning.chain_of_thought:  # do not replace with empty string
                    context["reasoning"] = reasoning.chain_of_thought
                    step.stream(reasoning.chain_of_thought, replace=True)
                    previous_plans.append(reasoning.chain_of_thought)

        return await self._fill_model(messages, reasoning.chain_of_thought, plan_model)

    async def _resolve_plan(
        self,
        raw_plan: RawPlan,
        agents: dict[str, Agent],
        tools: dict[str, Tool],
        messages: list[Message],
        context: TContext,
        previous_actors: list[str],
        is_followup: bool = False
    ) -> tuple[Plan, set[str], list[str]]:
        table_provided = False
        tasks = []
        provided = set(context)
        unmet_dependencies = set()
        steps = []
        actors = []
        actors_in_graph = set()

        # Check for consecutive agents of the same type and merge them
        merged_steps = []
        i = 0
        while i < len(raw_plan.steps):
            current_step = raw_plan.steps[i]

            # Look ahead to see if there are consecutive steps with the same actor
            consecutive_steps = [current_step]
            j = i + 1
            while j < len(raw_plan.steps) and raw_plan.steps[j].actor == current_step.actor:
                consecutive_steps.append(raw_plan.steps[j])
                j += 1

            if len(consecutive_steps) > 1:
                # Join the consecutive steps into one
                combined_instruction = f"{current_step.instruction} Additionally: " + ". ".join(step.instruction for step in consecutive_steps[1:])
                combined_title = f"{current_step.title} (combined {len(consecutive_steps)} steps)"

                # Create a new step with combined instruction
                merged_step = type(current_step)(
                    actor=current_step.actor,
                    instruction=combined_instruction,
                    title=combined_title,
                )
                merged_steps.append(merged_step)
            else:
                merged_steps.append(current_step)

            i = j

        # Update the raw_plan with merged steps
        raw_plan.steps = merged_steps

        for step in raw_plan.steps:
            key = step.actor
            actors.append(key)
            if key in agents:
                subagent = agents[key]
            elif key in tools:
                subagent = tools[key]
            else:
                # Skip if the agent or tool doesn't exist
                log_debug(f"Warning: Agent or tool '{key}' not found in available agents/tools")
                continue

            requires = set(await subagent.requirements(messages))
            provided |= set(subagent.output_schema.__annotations__)
            unmet_dependencies = (unmet_dependencies | requires) - provided
            has_table_lookup = any(isinstance(task.actor, MetadataLookup) for task in tasks)
            if "table" in unmet_dependencies and not table_provided and "SQLAgent" in agents and has_table_lookup:
                provided |= set(agents["SQLAgent"].output_schema.__annotations__)
                sql_step = type(step)(
                    actor="SQLAgent",
                    instruction="Load the table",
                    title="Loading table",
                )
                tasks.append(ActorTask(agents["SQLAgent"], instruction=sql_step.instruction, title=sql_step.title))
                steps.append(sql_step)
                table_provided = True
                unmet_dependencies -= provided
                actors_in_graph.add("SQLAgent")
            tasks.append(ActorTask(subagent, instruction=step.instruction, title=step.title))
            steps.append(step)
            actors_in_graph.add(key)

        last_task = tasks[-1] if tasks else None
        if last_task and isinstance(last_task.actor, Tool):
            if "AnalystAgent" in agents and all(r in provided for r in agents["AnalystAgent"].input_schema.__annotations__):
                actor = "AnalystAgent"
            else:
                actor = "ChatAgent"

            # Check if the actor conflicts with any actor in the graph
            not_with = getattr(agents[actor], "not_with", [])
            conflicts = [actor for actor in actors_in_graph if actor in not_with]
            if conflicts:
                # Skip the summarization step if there's a conflict
                log_debug(f"Skipping summarization with {actor} due to conflicts: {conflicts}")
                raw_plan.steps = steps
                previous_actors = actors
                plan = Plan(
                    *tasks, title=raw_plan.title, history=messages, context=context, coordinator=self, steps_layout=self.steps_layout, is_followup=is_followup
                )
                return plan, previous_actors

            summarize_step = type(step)(
                actor=actor,
                instruction="Summarize the results.",
                title="Summarizing results",
            )
            steps.append(summarize_step)
            tasks.append(
                ActorTask(
                    agents[actor],
                    instruction=summarize_step.instruction,
                    title=summarize_step.title,
                )
            )
            actors_in_graph.add(actor)

        if "ValidationAgent" in agents and len(actors_in_graph) > 1 and self.validation_enabled:
            validation_step = type(step)(
                actor="ValidationAgent",
                instruction="Validate whether the executed plan fully answered the user's original query.",
                title="Validating results",
            )
            steps.append(validation_step)
            tasks.append(ActorTask(agents["ValidationAgent"], instruction=validation_step.instruction, title=validation_step.title))
            actors_in_graph.add("ValidationAgent")

        raw_plan.steps = steps
        plan = Plan(
            *tasks, title=raw_plan.title, history=messages, context=context, coordinator=self, steps_layout=self.steps_layout, is_followup=is_followup
        )
        return plan, actors

    async def _compute_plan(
        self, messages: list[Message], context: TContext, agents: dict[str, Agent], tools: dict[str, Tool], pre_plan_output: dict[str, Any]
    ) -> Plan:
        tool_names = list(tools)
        agent_names = list(agents)
        plan_model = self._get_model("main", agents=agent_names, tools=tool_names)
        is_followup = pre_plan_output["is_follow_up"]

        planned = False
        unmet_dependencies = set()
        previous_plans, previous_actors = [], []
        attempts = 0
        plan = None
        with self._add_step(title="Planning how to solve user query...", user="Planner", steps_layout=self.steps_layout) as istep:
            await asyncio.sleep(0.1)
            while not planned:
                if attempts > 0:
                    log_debug(f"\033[91m!! Attempt {attempts}\033[0m")
                plan = None
                try:
                    raw_plan = await self._make_plan(
                        messages,
                        context,
                        agents,
                        tools,
                        unmet_dependencies,
                        previous_actors,
                        previous_plans,
                        plan_model,
                        istep,
                        is_follow_up=is_followup,
                    )
                except asyncio.CancelledError as e:
                    self._todos_title.object = istep.failed_title = "Planning was cancelled, please try again."
                    traceback.print_exception(e)
                    raise e
                except Exception as e:
                    istep.failed_title = "Internal execution error during planning stage."
                    self._todos_title.object = "Planner could not settle on a plan of action to perform the requested query. Please restate your request."
                    traceback.print_exception(e)
                    raise e
                plan, previous_actors = await self._resolve_plan(
                    raw_plan, agents, tools, messages, context, previous_actors, is_followup=is_followup
                )
                try:
                    plan.validate()
                except ContextError as e:
                    istep.stream(str(e), replace=True)
                    attempts += 1
                else:
                    planned = True
                if attempts > 5:
                    istep.failed_title = "Planning failed to come up with viable plan, please restate the problem and try again."
                    self._todos_title.object = "❌ Planning failed"
                    e = RuntimeError("Planner failed to come up with viable plan after 5 attempts.")
                    traceback.print_exception(e)
                    raise e

            # Store the todo message reference for later updates
            if attempts > 0:
                istep.success_title = f"Plan with {len(raw_plan.steps)} steps created after {attempts + 1} attempts"
            else:
                istep.success_title = f"Plan with {len(raw_plan.steps)} steps created"
        return plan
