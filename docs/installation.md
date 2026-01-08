# :material-download: Installation

Lumen works with Python 3.11+ on Linux, Windows, and Mac.

!!! tip "Already installed?"
    Jump to the [Quick Start](quick_start.md) to start chatting with your data.

## Bring Your Own LLM

Lumen works with any LLM provider. Choose the approach that fits your needs:

- ☁️ **Cloud providers** — OpenAI, Anthropic, Google, Mistral, Azure (easiest to get started)
- 🖥️ **Locally hosted** — Ollama, Llama.cpp (free, runs on your machine, no API keys)
- 🔀 **Router/Multi-provider** — LiteLLM (unified interface, 100+ models)

---

## Cloud Service Providers

Use hosted LLM APIs from major providers. Fastest to set up, pay-per-use pricing.

=== "OpenAI"

    ```bash
    pip install 'lumen[ai-openai]'
    export OPENAI_API_KEY=sk-...
    ```

    Get your API key from [platform.openai.com](https://platform.openai.com/api-keys)

=== "Anthropic"

    ```bash
    pip install 'lumen[ai-anthropic]'
    export ANTHROPIC_API_KEY=sk-ant-...
    ```

    Get your API key from [console.anthropic.com](https://console.anthropic.com/account/keys)

=== "Google Gemini"

    ```bash
    pip install 'lumen[ai-google]'
    export GEMINI_API_KEY=your-key
    ```

    Get your API key from [AI Studio](https://aistudio.google.com/app/apikey)

=== "Mistral"

    ```bash
    pip install 'lumen[ai-mistralai]'
    export MISTRAL_API_KEY=your-key
    ```

    Get your API key from [console.mistral.ai](https://console.mistral.ai/api-keys/)

=== "Azure OpenAI"

    ```bash
    pip install 'lumen[ai-openai]'
    export AZURE_OPENAI_API_KEY=your-key
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_API_VERSION=2024-02-15-preview
    ```

    Configure your Azure OpenAI resource in the [Azure Portal](https://portal.azure.com/)

---

## Locally Hosted

Run open-source LLMs on your own machine. No API keys required, full privacy, free to use.

=== "Ollama"

    ```bash
    pip install 'lumen[ai-ollama]'
    ```

    1. Install [Ollama](https://ollama.ai)
    2. Download a model: `ollama pull llama2`
    3. Start Ollama: `ollama serve`
    4. Optional: Set custom endpoint if not using default
    
    ```bash
    export OLLAMA_BASE_URL=http://localhost:11434
    ```

=== "Llama.cpp"

    ```bash
    pip install 'lumen[ai-llama]'
    ```

    1. Install [llama.cpp](https://github.com/ggerganov/llama.cpp)
    2. Download a GGUF model
    3. Start the server: `./server -m model.gguf`
    4. Configure the endpoint if needed

---

## Router / Multi-Provider

Use a unified interface to switch between multiple LLM providers. Useful for testing different models or supporting multiple backends.

=== "LiteLLM"

    ```bash
    pip install 'lumen[ai-litellm]'
    ```

    Supports 100+ models across OpenAI, Anthropic, Google, Mistral, and more.

    Set environment variables for your providers:

    ```bash
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=...
    ```

    Then configure your model in Lumen using the provider prefix:

    ```python
    # In your Lumen config
    llm_provider: "gpt-4"          # OpenAI
    # or
    llm_provider: "claude-3-opus"  # Anthropic
    # or
    llm_provider: "gemini-pro"     # Google
    ```

---

## Verify Installation

```bash
lumen-ai --version
```

## Next Steps

Ready to start? Head to the [Quick Start](quick_start.md) guide to chat with your first dataset.
