"""
Base class for agents that execute LLM-generated code.

This provides a clean inheritance hierarchy:
    BaseViewAgent
        └── BaseCodeAgent
                ├── VegaLiteAgent
                └── (future code-executing agents)
"""
from __future__ import annotations

import asyncio

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import param

from panel.layout import Row
from panel_material_ui import Button, Details, Markdown

from ..code_executor import CodeExecutor
from .base_view import BaseViewAgent

if TYPE_CHECKING:
    from pandas import DataFrame
    from panel.chat.step import ChatStep

    from ...pipeline import Pipeline
    from ..context import TContext
    from ..llm import Message


class BaseCodeAgent(BaseViewAgent):
    """
    Base class for view agents that can generate and execute code.

    Subclasses must:
    - Set `_executor_class` to the appropriate CodeExecutor subclass
    - Implement `_generate_code_spec()` for their specific code generation flow
    """

    code_execution = param.Selector(
        default="disabled",
        objects=["disabled", "prompt", "llm", "allow"],
        doc="""
        Code execution mode for generating visualizations via code:
        - disabled: No code execution; generate declarative specs only (safe for production)
        - prompt: Generate code, prompt user for permission to execute
        - llm: Generate code, validate with LLM safety check, then execute
        - allow: Generate and execute code without user confirmation

        ⚠️ WARNING: The 'prompt', 'llm', and 'allow' modes execute LLM-generated code and
        must NEVER be enabled in production environments with access to secrets, credentials,
        or sensitive data.
        """,
        allow_refs=True
    )

    # Subclasses MUST override this with the appropriate executor class
    _executor_class: type[CodeExecutor] = None

    # --------------------------------------------------------------------------
    # Abstract methods - subclasses must implement
    # --------------------------------------------------------------------------

    @abstractmethod
    async def _generate_code_spec(self, *args, **kwargs) -> dict[str, Any] | None:
        """Generate a view specification by executing LLM-generated code.

        Subclasses implement their full code generation flow here, including:
        - Rendering prompts and calling the LLM
        - Extracting code from the response
        - Calling `_execute_code()` to run it
        - Converting the result to a spec dict

        Returns
        -------
        dict[str, Any] | None
            The view specification, or None if execution was rejected.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------
    # Code validation and execution
    # --------------------------------------------------------------------------

    def _validate_code(self, code: str) -> list[str]:
        """Validate code for safety using AST inspection.

        Returns
        -------
        list[str]
            List of validation errors. Empty list means code passed validation.
        """
        if self._executor_class is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must set _executor_class"
            )
        return self._executor_class.validate(code)

    async def _validate_code_with_llm(
        self,
        code: str,
        system_prompt: str,
    ) -> tuple[bool, str]:
        """Validate code safety using LLM inspection.

        Returns
        -------
        tuple[bool, str]
            (is_safe, reasoning)
        """
        if self._executor_class is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must set _executor_class"
            )
        return await self._executor_class.validate_with_llm(
            code, self.llm, system_prompt, self.llm_spec_key
        )

    async def _prompt_user_for_execution(self, code: str, language: str = "python") -> bool:
        """Prompt the user to approve or reject code execution.

        Returns
        -------
        bool
            True if user approved, False if rejected.
        """
        if self.interface is None:
            return True

        approval_event = asyncio.Event()

        code_display = Markdown(
            f"```{language}\n{code}\n```",
            margin=0,
            styles={"width": "100%", "max-width": "100%"},
            stylesheets=['.codehilite { margin: 0; border-radius: 0; } div { width: 100%; }'],
        )
        reject = Button(
            icon="cancel",
            label="Reject",
            on_click=lambda _: approval_event.set(),
            variant="outlined"
        )
        accept = Button(
            description="⚠️ Confirm generated code is safe to execute.",
            description_delay=100,
            icon="check",
            label="Accept",
            on_click=lambda _: approval_event.set()
        )
        prompt_content = Details(
            code_display,
            collapsed=False,
            header=Row(
                "### Confirm Code Execution",
                Row(reject, accept, styles={"margin": "0 0 0 auto"}),
                sizing_mode="stretch_width"
            ),
            scrollable_height=300,
            stylesheets=[".message { margin-left: 0; padding-inline: 0}"],
            sizing_mode="stretch_width",
        )
        self.interface.send(prompt_content, respond=False, user="Assistant")
        await approval_event.wait()
        accepted = accept.clicks > 0
        prompt_content.header = f"### Code Execution {'Accepted' if accepted else 'Rejected'}"
        return accepted

    async def _execute_code(
        self,
        code: str,
        df: DataFrame,
        system: str | None = None,
        step: ChatStep | None = None,
    ) -> Any | None:
        """Validate and execute code based on the code_execution mode.

        This method handles the full execution flow:
        1. AST validation (always)
        2. LLM validation (if code_execution="llm")
        3. User prompting (if code_execution="prompt")
        4. Code execution

        Parameters
        ----------
        code : str
            The code to execute
        df : DataFrame
            Data available as `df` in the code
        system : str | None
            System prompt for LLM validation (required if code_execution="llm")
        step : ChatStep | None
            Optional chat step to stream messages to

        Returns
        -------
        Any | None
            The execution result, or None if user rejected execution.

        Raises
        ------
        ValueError
            If code_execution is "disabled" or code fails validation.
        """
        if self._executor_class is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must set _executor_class"
            )

        if self.code_execution == "disabled":
            raise ValueError("Code execution is disabled")

        # Step 1: AST validation (always)
        validation_errors = self._validate_code(code)
        if validation_errors:
            error_msg = "; ".join(validation_errors)
            raise ValueError(f"Unsafe code detected: {error_msg}")

        # Step 2: LLM validation (if mode is "llm")
        if self.code_execution == "llm":
            if system is None:
                raise ValueError("system is required when code_execution='llm'")
            is_safe, reasoning = await self._validate_code_with_llm(code, system)
            if step:
                step.stream(f"LLM decided the code is {'safe' if is_safe else 'unsafe'}: {reasoning}")
            if not is_safe:
                raise ValueError(f"LLM security review failed: {reasoning}")

        # Step 3: User prompt (if mode is "prompt")
        if self.code_execution == "prompt":
            approved = await self._prompt_user_for_execution(code)
            if step:
                step.stream(f"User {'approved' if approved else 'rejected'} code execution.")
            if not approved:
                return None

        # Step 4: Execute
        return self._executor_class.execute(code, df)

    @property
    def code_execution_enabled(self) -> bool:
        """Whether code execution is enabled (any mode except 'disabled')."""
        return self.code_execution != "disabled"

    async def _generate_spec(
        self,
        messages: list[Message],
        context: TContext,
        pipeline: Pipeline,
        doc: str,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Generate a view specification using either code execution or declarative mode.

        Delegates to `_generate_code_spec()` or `_generate_yaml_spec()` based on
        the `code_execution` setting.

        Parameters
        ----------
        messages : list[Message]
            The conversation messages
        context : TContext
            The context dictionary
        pipeline : Pipeline
            The data pipeline
        doc : str
            Documentation string for the view type
        **kwargs
            Additional arguments passed to the generation methods

        Returns
        -------
        dict[str, Any] | None
            The view specification, or None if code execution was rejected.
        """
        if self.code_execution_enabled:
            return await self._generate_code_spec(messages, context, pipeline, doc, **kwargs)
        else:
            return await self._generate_yaml_spec(messages, context, pipeline, doc, **kwargs)
