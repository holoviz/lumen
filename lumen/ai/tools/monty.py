from .base import FunctionTool


def make_monty_llm_tool() -> FunctionTool:
    async def run_python(code: str) -> str:
        """
        Run a short Python snippet in an isolated Monty sandbox.

        Monty implements a subset of Python, not CPython. Only its bundled
        standard-library modules can be imported; third-party packages such as
        pandas and numpy are unavailable. The sandbox has no access to Lumen's
        variables, files, environment, network, or subprocesses. Each call starts
        with fresh state, so pass all data in the snippet and return a value as
        the last expression or use print(). Keep calculations small: execution
        time, memory, and printed output are limited.

        Parameters
        ----------
        code : str
            Python source to execute. The final expression and printed output
            are returned to the agent.
        """
        from pydantic_monty import AsyncMonty, CollectString, MontyError

        output = CollectString(max_bytes=65_536)
        try:
            async with AsyncMonty(request_timeout=3) as pool:
                async with pool.checkout(limits={"max_memory": 10_000_000, "max_feed_duration_secs": 1.0}) as session:
                    result = await session.feed_run(code, print_callback=output)
        except MontyError as exc:
            return f"{output.output}Error: {exc}"
        if result is not None:
            return (output.output + repr(result))[:65_536]
        return output.output or "No output"

    return FunctionTool(run_python)
