from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import traceback

from textwrap import dedent

import bokeh.command.subcommands.serve

from bokeh.application.handlers.code import CodeHandler
from bokeh.command.util import die
from panel.command import Serve, transform_cmds
from panel.io.application import Application

try:
    from lumen.ai.config import THIS_DIR
except ImportError:
    print('You need to install lumen-ai with "lumen[ai]"')
    sys.exit(1)

from ..ai import agents as lumen_agents  # Aliased here
from ..ai.utils import render_template

CMD_DIR = THIS_DIR / ".." / "command"

LLM_PROVIDERS = {
    'openai': 'OpenAI',
    'anthropic': 'AnthropicAI',
    'mistral': 'MistralAI',
    'azure-openai': 'AzureOpenAI',
    'azure-mistral': 'AzureMistralAI',
    "ai-navigator": "AINavigator",
    'llama': 'Llama',
}


class LLMConfig:
    """Configuration handler for LLM providers"""

    PROVIDER_ENV_VARS = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "azure-mistral": "AZUREAI_ENDPOINT_KEY",
        "azure-openai": "AZUREAI_ENDPOINT_KEY",
    }

    @classmethod
    def detect_provider(cls) -> str | None:
        """Detect available LLM provider based on environment variables"""
        for provider, env_var in cls.PROVIDER_ENV_VARS.items():
            if env_var and os.environ.get(env_var):
                return provider
        return None


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
            "--model-kwargs",
            type=str,
            help="JSON string of model keyword arguments for the LLM. Example: --model-kwargs '{\"repo\": \"abcdef\"}'",
        )

    def invoke(self, args: argparse.Namespace) -> bool:
        """Override invoke to handle both sets of arguments"""
        provider = args.provider
        api_key = args.api_key
        endpoint = args.provider_endpoint
        mode = args.validation_mode
        temperature = args.temperature
        agents = args.agents
        log_level = args.log_level

        if not provider:
            provider = LLMConfig.detect_provider()

        model_kwargs = None
        if args.model_kwargs:
            try:
                model_kwargs = json.loads(args.model_kwargs)
            except json.JSONDecodeError as e:
                die(f"Invalid JSON format for --model-kwargs: {e}\n"
                    f"Ensure the argument is properly escaped. Example: --model-kwargs '{{\"key\": \"value\"}}'")

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
        **kwargs,
    ) -> None:
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
        )
        super().__init__(filename="lumen_ai.py", source=source, **kwargs)

    def _build_source_code(self, tables: list[str], **config) -> str:
        """Build source code with configuration"""
        if config.get("provider") is None:
            raise RuntimeError(
                "No LLM provider was detected. To resolve this specify a provider "
                "using the --provider CLI argument along with an API key (using --api-key) "
                "or configure environment variable(s) for your favorite LLM cloud provider "
                "(e.g. OPENAI_API_KEY or ANTHROPIC_API_KEY). If you are running an OpenAI "
                "compatible API specify a custom endpoint with --provider-endpoint."
            )

        context = {
            "llm_provider": LLM_PROVIDERS[config['provider']],
            "tables": [repr(t) for t in tables],
            "api_key": config.get("api_key"),
            "endpoint": config.get("endpoint"),
            "mode": config.get("mode"),
            "agents": config.get("agents"),
            "log_level": config["log_level"],
            "temperature": config.get("temperature"),
            "model_kwargs": config.get('model_kwargs') or {},
        }
        context = {k: v for k, v in context.items() if v is not None}

        source = render_template(
            CMD_DIR / "app.py.jinja2", relative_to=CMD_DIR, **context
        ).replace("\n\n", "\n").strip()
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
        if levels.get(args.log_level.upper(), 30) < 20:
            traceback.print_exc()
        die("ERROR: " + str(e))

    if ret is False:
        sys.exit(1)
    elif ret is not True and isinstance(ret, int) and ret != 0:
        sys.exit(ret)


if __name__ == "__main__":
    main()
