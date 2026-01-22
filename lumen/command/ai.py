from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import traceback

from pathlib import Path
from textwrap import dedent

import bokeh.command.subcommands.serve

from bokeh.application.handlers.code import CodeHandler
from bokeh.command.util import die
from panel.command import Serve, transform_cmds
from panel.io.application import Application

try:
    from ..ai.config import THIS_DIR
except ImportError as e:
    print(f'You need to install lumen-ai with "pip install lumen[ai]": {e}')  # noqa: T201
    sys.exit(1)

from ..ai import agents as lumen_agents, llm as lumen_llms  # Aliased here
from ..ai.llm import LLM_PROVIDERS, get_available_llm
from ..ai.utils import parse_huggingface_url, render_template

CMD_DIR = THIS_DIR / ".." / "command"


class LumenAIServe(Serve):
    """Extended Serve command that handles both Panel/Bokeh and Lumen AI arguments"""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__(parser=parser)
        self.add_lumen_arguments(parser)

    def add_lumen_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add Lumen AI specific arguments to the parser"""
        group = parser.add_argument_group("Lumen AI Configuration")
        group.add_argument(
            "--provider",
            choices=list(LLM_PROVIDERS.keys()),
            help="LLM provider (auto-detected from environment variables if not specified)",
        )
        group.add_argument("--api-key", help="API key for the LLM provider")
        group.add_argument("--provider-endpoint", help="Custom endpoint for the LLM provider")
        group.add_argument("--validation-mode", help="Validation mode for the LLM")
        group.add_argument("--temperature", type=float, help="Temperature for the LLM")
        group.add_argument("--agents", nargs="+", help="Additional agents to include")
        group.add_argument(
            "--model",
            type=str,
            help="Model name to use (sets model in model_kwargs['default']). Example: --model 'qwen3:8b'",
        )
        group.add_argument(
            "--model-kwargs",
            type=str,
            help="JSON string of model keyword arguments for the LLM. Example: --model-kwargs '{\"default\": {\"repo\": \"abcdef\"}}'",
        )
        group.add_argument(
            "--llm-model-url",
            type=str,
            help="""
            Huggingface URL to the GGUF file and model kwargs as query params.
            Example --llm-model-url 'https://huggingface.co/RE/PO/blob/main/FILE.gguf?chat_format=chat_format'
            """,
        )
        group.add_argument(
            "--logfire-tags",
            nargs="*",
            help="Tags for logfire logging. If provided, enables logfire logging with the specified tags.",
        )
        group.add_argument(
            "--code-execution",
            choices=["hidden", "disabled", "prompt", "llm", "bypass"],
            default="hidden",
            help="""
            Code execution mode for generating Vega-Lite specs via Altair code.
            Controls whether the code execution selector appears in the UI preferences:
            - hidden: Do not show code execution option in preferences (default)
            - disabled: Show selector, but default to no code execution (Vega-Lite spec only)
            - prompt: Show selector, default to prompting user for permission to execute
            - llm: Show selector, default to LLM-validated code execution
            - allow: Show selector, default to executing code without confirmation

            WARNING: The 'prompt', 'llm', and 'allow' modes execute LLM-generated code and
            must NEVER be enabled in production environments with access to secrets, credentials,
            or sensitive data.
            """,
        )

    def invoke(self, args: argparse.Namespace) -> bool:
        """Override invoke to handle both sets of arguments"""
        provider = args.provider
        llm_model_url = args.llm_model_url
        provider_cls = None
        if llm_model_url and provider and provider != "llama":
            raise ValueError(
                f"Cannot specify both --llm-model-url and --provider {provider!r}. "
                f"Use --llm-model-url to load a model from HuggingFace."
            )
        elif llm_model_url:
            provider = "llama"
        elif not provider:
            provider_cls = get_available_llm()

        if provider is None and provider_cls is None:
            raise RuntimeError(
                "It looks like a Language Model provider isn't set up yet.\n"
                "You have a few options to resolve this:\n\n"
                "- Set environment variables with an API key: For example, OPENAI_API_KEY or ANTHROPIC_API_KEY.\n"
                "- Specify a provider and API key directly: For example, set `--provider openai` with your API key via --api-key.\n"
                "- Custom endpoint: If using an OpenAI-compatible API, set --provider openai and define the --provider-endpoint.\n\n"
                "If you still need assistance visit the docs: https://lumen.holoviz.org/lumen_ai/how_to/llm/index.html"
            )

        api_key = args.api_key
        endpoint = args.provider_endpoint
        mode = args.validation_mode
        temperature = args.temperature
        agents = args.agents
        log_level = args.log_level
        logfire_tags = getattr(args, 'logfire_tags', None)
        model = getattr(args, 'model', None)
        code_execution = getattr(args, 'code_execution', 'hide')

        if provider_cls is None:
            try:
                provider_cls = getattr(lumen_llms, LLM_PROVIDERS[provider])
            except (KeyError, AttributeError) as err:
                raise ValueError(
                    f"Could not find LLM Provider {provider!r}, valid providers include: {list(LLM_PROVIDERS)}."
                ) from err

        model_kwargs = None
        if args.model_kwargs or llm_model_url or model:
            model_kwargs = {}
            if args.model_kwargs:
                try:
                    model_kwargs = json.loads(args.model_kwargs)
                except json.JSONDecodeError as e:
                    die(f"Invalid JSON format for --model-kwargs: {e}\n"
                        f"Ensure the argument is properly escaped. Example: --model-kwargs '{{\"key\": \"value\"}}'")

            # Handle --model argument
            if model:
                if 'default' not in model_kwargs:
                    model_kwargs['default'] = {}
                model_kwargs['default']['model'] = model

            # Handle huggingface URL if provided
            if llm_model_url:
                try:
                    repo, model_file, query_kwargs = parse_huggingface_url(llm_model_url)
                    # Initialize default dict if not exists
                    if 'default' not in model_kwargs:
                        model_kwargs['default'] = {}
                    model_kwargs['default'].update({
                        'repo': repo,
                        'model_file': model_file,
                        **query_kwargs
                    })
                except ValueError as e:
                    die(str(e))
        provider_cls.warmup(model_kwargs)

        agent_classes = [
            (name, cls) for name, cls in inspect.getmembers(lumen_agents, inspect.isclass)
            if issubclass(cls, lumen_agents.Agent) and cls is not lumen_agents.Agent
        ]
        agent_class_names = {name.lower(): name for name, cls in agent_classes}

        if agents:
            # Adjust agent names to match the class names, case-insensitively
            agents = [
                agent_class_names.get(agent.lower()) or
                agent_class_names.get(f"{agent.lower()}agent") or
                agent
                for agent in agents
            ]

        def build_single_handler_applications(
            paths: list[str], argvs: dict[str, list[str]] | None = None
        ) -> dict[str, Application]:
            kwargs = dict(
                provider=provider,
                api_key=api_key,
                temperature=temperature,
                endpoint=endpoint,
                mode=mode,
                agents=agents,
                log_level=log_level,
                model_kwargs=model_kwargs,
                logfire_tags=logfire_tags,
                code_execution=code_execution,
            )
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            handler = AIHandler(paths, **kwargs)
            if handler.failed:
                raise RuntimeError(
                    f"Error loading {paths}:\n\n{handler.error}\n{handler.error_detail}"
                )
            return {"/lumen_ai": Application(handler)}

        bokeh.command.subcommands.serve.build_single_handler_applications = (
            build_single_handler_applications
        )

        return super().invoke(args)


class AIHandler(CodeHandler):
    """Handler for Lumen AI applications"""

    def __init__(
        self,
        tables: list[str],
        provider: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        endpoint: str | None = None,
        mode: str | None = None,
        agents: list[str] | None = None,
        log_level: str = "INFO",
        model_kwargs: dict | None = None,
        logfire_tags: list[str] | None = None,
        code_execution: str = "hidden",
        **kwargs,
    ) -> None:
        for table_path in tables:
            if table_path == "no_data":
                continue

            # Skip validation for non-local sources (e.g. URLs, DB connection strings)
            if "://" in table_path or table_path.startswith(
                ("sqlite:", "postgresql:", "mysql:", "mssql:", "oracle:", "duckdb:")
            ):
                continue

            path = Path(table_path)
            if not path.is_file():
                if path.exists():
                    raise ValueError(f"Table path is not a file: {table_path}")
                raise FileNotFoundError(f"Table file not found: {table_path}")

        source = self._build_source_code(
            tables=tables,
            provider=provider,
            api_key=api_key,
            temperature=temperature,
            endpoint=endpoint,
            mode=mode,
            agents=agents,
            log_level=log_level,
            model_kwargs=model_kwargs,
            logfire_tags=logfire_tags,
            code_execution=code_execution,
        )
        super().__init__(filename="lumen_ai.py", source=source, **kwargs)

    def _build_source_code(self, tables: list[str], **config) -> str:
        """Build source code with configuration"""
        # Only include llm_provider in context if a provider was explicitly specified
        provider = config.get('provider')

        context = {
            "tables": [repr(t) for t in tables],
            "api_key": config.get("api_key"),
            "endpoint": config.get("endpoint"),
            "mode": config.get("mode"),
            "agents": config.get("agents"),
            "log_level": config["log_level"],
            "temperature": config.get("temperature"),
            "model_kwargs": config.get('model_kwargs') or {},
            "logfire_tags": config.get("logfire_tags"),
            "code_execution": config.get("code_execution"),
        }

        # Only add llm_provider if explicitly specified
        if provider is not None:
            if provider not in LLM_PROVIDERS:
                available_providers = list(LLM_PROVIDERS.keys())
                raise ValueError(
                    f"Unknown provider '{provider}'. Available providers: {available_providers}"
                )
            context["llm_provider"] = LLM_PROVIDERS[provider]

        context = {k: v for k, v in context.items() if v is not None}

        source = render_template(
            CMD_DIR / "app.py.jinja2", relative_to=CMD_DIR, **context
        ).replace("\n\n", "\n").strip()

        print(f"\033[93mGenerated source code:\n```python\n{source}\n```\033[0m\n")  # noqa: T201 for reusability
        return source


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="lumen-ai",
        description=dedent("""\
            Lumen AI - Launch Lumen AI applications with customizable LLM configuration.
            To start the application without any data, simply run 'lumen-ai' with no additional arguments.

            First time running Lumen AI take a look at getting started documentation:
            https://lumen.holoviz.org/lumen_ai/getting_started/

            Found a Bug or Have a Feature Request?
            Open an issue at: https://github.com/holoviz/lumen/issues

            Have a Question?
            Ask on our Discord chat server: https://discord.gg/rb6gPXbdAr

            Need Help?
            Ask a question on our forum: https://discourse.holoviz.org
        """),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="See '<command> --help' to read about a specific subcommand.",
    )

    parser.add_argument("-v", "--version", action="version", version="Lumen AI 1.0.0")

    subs = parser.add_subparsers(help="Sub-commands", dest="command")

    serve_parser = subs.add_parser(
        Serve.name, help="Run a bokeh server to serve the Lumen AI application."
    )
    serve_command = LumenAIServe(parser=serve_parser)
    serve_parser.set_defaults(invoke=serve_command.invoke)

    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        args = parser.parse_args(sys.argv[1:])
        args.invoke(args)
        sys.exit()

    if len(sys.argv) == 1:
        sys.argv.extend(["serve", "no_data"])

    sys.argv = transform_cmds(sys.argv)
    args = parser.parse_args(sys.argv[1:])

    if not hasattr(args, "invoke"):
        parser.print_help()
        sys.exit(1)

    try:
        ret = args.invoke(args)
    except Exception as e:
        levels = logging.getLevelNamesMapping()
        if levels.get((args.log_level or 'warn').upper(), 30) < 20:
            traceback.print_exc()
        die("ERROR: " + str(e))

    if ret is False:
        sys.exit(1)
    elif ret is not True and isinstance(ret, int) and ret != 0:
        sys.exit(ret)


if __name__ == "__main__":
    main()
