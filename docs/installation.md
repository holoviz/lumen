# Installation

Lumen works with Python 3.10+ on Linux, Windows, and Mac.

## Bring Your Own LLM

Lumen works with any LLM provider. No vendor lock-in.

- 🔓 **Open models** — Ollama, Llama.cpp (free, runs locally)
- ☁️ **Cloud providers** — OpenAI, Anthropic, Google, Mistral, Azure

---

## Install Lumen with AI

=== "OpenAI"

    ```bash
    pip install 'lumen[ai-openai]'
    export OPENAI_API_KEY=sk-...
    ```

=== "Anthropic"

    ```bash
    pip install 'lumen[ai-anthropic]'
    export ANTHROPIC_API_KEY=sk-ant-...
    ```

=== "Google Gemini"

    ```bash
    pip install 'lumen[ai-google]'
    export GEMINI_API_KEY=your-key
    ```

=== "Mistral"

    ```bash
    pip install 'lumen[ai-mistralai]'
    export MISTRAL_API_KEY=your-key
    ```

=== "Azure OpenAI"

    ```bash
    pip install 'lumen[ai-azure]'
    export AZURE_OPENAI_API_KEY=your-key
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_API_VERSION=2024-02-15-preview
    ```

=== "Local LLM (Llama.cpp)"

    ```bash
    pip install 'lumen[ai-llama]'
    ```

=== "Ollama"

    ```bash
    pip install 'lumen[ai-ollama]'
    ```

    Make sure Ollama is running locally. Optional: set the endpoint if not using default.

    ```bash
    export OLLAMA_BASE_URL=http://localhost:11434
    ```

=== "LiteLLM (Multi-provider)"

    ```bash
    pip install 'lumen[ai-litellm]'
    ```

    Supports 100+ models across OpenAI, Anthropic, Google, Mistral, and more. Configure your provider's API key as needed.

---

## Verify Installation

```bash
lumen --version
```
