# Custom Agents

Agents in Lumen AI extend the system's capabilities by taking on specialized roles to solve sub-tasks, interpret user queries, and produce structured outputs. Built-in Agents support common workflows, such as querying datasets, creating visualizations, and generating analyses. Creating custom Agents allows fine-tuning their behavior, domain knowledge, and workflow to fit specific use cases.

This guide covers how to:

- Subclass the `Agent` class to define custom behavior and purpose.
- Specify the context your Agent requires or provides.
- Integrate your Agent into the system to address user queries.

## Key Concepts

**`requires`**:
Defines the memory context an Agent depends on to function. For instance, an `SQLAgent` might `requires=["source"]` to generate a query based on a dataset.

**`provides`**:
Defines the context an Agent updates or adds in memory after completing its task. For example, a `SQLAgent` might `provides=["sql"]` to store the generated SQL query for later use.

**`purpose`**:
A description guiding the LLM on when to invoke the Agent. This ensures each the `Coordinator` can select the right `Agent` for a given task.

## Building Agents

1. **Subclass the `Agent` class:**
   Inherit from `Agent` to access its framework for memory, prompts, and UI integration.

2. **Define `requires` and `provides`:**
   Specify the context your Agent depends on (e.g., `requires=["source"]`) and what it outputs (e.g., `provides=["summary"]`).

3. **Set the `purpose` and `prompts`:**
   Use the `purpose` attribute to describe the Agent’s role. Define `prompts` to guide the LLM’s logic using templates for consistent responses.

4. **Implement the `respond` method:**
   Write the logic for handling queries, accessing memory, and producing outputs. The `respond` method can return text, data structures, or UI elements.

5. **Add the Agent to the system:**
   Register the custom Agent with `ExplorerUI` or a similar interface to enable it in user workflows.

## Example: Custom SummarizerAgent

```python
import param

class SummarizerAgent(lmai.agents.Agent):

    purpose = param.String(default="Summarizes documents for a high-level overview.")

    prompts = param.Dict(
        default={
            "main": {
                "template": "Please summarize the following document sources: {sources_texts}",
            },
        }
    )

    requires = param.List(default=["document_sources"])

    provides = param.List(default=["summary"])

    async def respond(self, messages: list[dict[str, str]], render_output: bool = True, **kwargs):
        sources = self._memory["document_sources"]
        sources_texts = ""
        for source in sources:
            sources_texts += "#" + source.get("metadata", "") + "\n"
            text = source["text"]
            sources_texts += text + "\n\n"
        system = await self._render_prompt("main", messages, sources_texts=sources_texts)
        summary = await self._stream(messages, system)
        self._memory["summary"] = summary
        return summary

lmai.ExplorerUI(agents=[SummarizerAgent()])
```

## Best Practices

- Start with single-task Agents; expand their complexity as needed.
- Provide clear `purpose` and structured prompts for consistency.
- Use `requires` and `provides` effectively to manage memory.

By following these guidelines, you can design Agents that integrate seamlessly with Lumen AI, enabling complex reasoning and user interaction.
