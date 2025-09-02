# Use Ollama with Lumen AI

Lumen AI ships with the {py:class}`lumen.ai.llm.Ollama` wrapper for Ollama local LLM models. Ollama enables you to run large language models locally, including Llama 3.3, DeepSeek-R1, Qwen 3, Mistral, Gemma 3, and many other models. Lumen picks between a default and a reasoning model to solve different tasks, by default it will invoke the following models:

- **`default`**: qwen2.5-coder:7b
- **`reasoning`**: qwen2.5-coder:7b (uses default since no reasoning model specified)

For information on configuring the models being used find the [configuring LLMs how-to guide](configure.md).

## Prerequisites

- Lumen AI installed in your Python environment.
- Ollama installed and running on your system. You can install Ollama from [ollama.com](https://ollama.com/).
- At least one model pulled locally. Use `ollama pull <model-name>` to download models.

## Installing and Setting Up Ollama

1. **Install Ollama**

Install Ollama using the official installation script:

::::{tab-set}

:::{tab-item} Unix/Linux/macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
:::

:::{tab-item} Windows
Download and run the installer from [ollama.com](https://ollama.com/)
:::

::::

2. **Start the Ollama Service**

```bash
ollama serve
```

3. **Pull Some Models**

Download the default model used by Lumen AI:

```bash
ollama pull qwen2.5-coder:7b
```

You can also explore other available models like DeepSeek-R1, Llama 3.3, Qwen3, Mistral Small 3.1, Gemma 3, and more from the Ollama library.

## Using with Default Configuration

Once Ollama is running with models available, simply run your Lumen AI application:

```bash
lumen-ai serve <your-data-file-or-url> --provider ollama
```

Lumen AI will automatically connect to Ollama running on `localhost:11434` (the default Ollama port).

## Using CLI Arguments

You can also provide the API endpoint as a CLI argument (though the default should work for most cases):

```bash
lumen-ai serve <your-data-file-or-url> --provider ollama --endpoint http://localhost:11434/v1
```

## Using Python

In Python, simply import the LLM wrapper {py:class}`lumen.ai.llm.Ollama` and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

ollama_llm = lmai.llm.Ollama(endpoint='http://localhost:11434/v1')

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=ollama_llm)
ui.servable()
```

## Configuring Models

If you want to override the `default` model, which uses `qwen2.5-coder:7b`, or add a `reasoning` model, a model configuration can be provided via the `model_kwargs` parameter. Lumen will use different models for different scenarios, currently a `default` and a `reasoning` model can be provided. If no reasoning model is provided it will always use the `default` model.

You can find available models in the Ollama library. To swap `qwen2.5-coder:7b` for `gemma3:2b` as the default model and add `deepseek-r1:8b` for reasoning:

```python
import lumen.ai as lmai

config = {
    "default": {"model": "gemma3:2b"},
    "reasoning": {"model": "deepseek-r1:8b"}
}

llm = lmai.llm.Ollama(model_kwargs=config, endpoint='http://localhost:11434/v1')

lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm).servable()
```

## Popular Model Recommendations

Based on current performance benchmarks and developer preferences in 2025, here are some recommended models for different use cases:

### General Purpose
- **gemma3:2b** - Google's latest Gemma 3 models designed for efficient execution on everyday devices
- **llama3.3:70b** - New state of the art 70B model with similar performance to Llama 3.1 405B
- **qwen3:7b** - Latest generation Qwen model with comprehensive suite of capabilities
- **qwen2.5:7b** - Excellent multilingual support with 128K context window

### Lightweight & Efficient Options
- **gemma3:2b** - High-performing and efficient model from Google designed for phones/tablets
- **smollm2:135m** - Ultra-lightweight model for resource-constrained environments
- **phi-4:14b** - Microsoft's lightweight state-of-the-art open model
- **tinyllama:1.1b** - Compact 1.1B Llama model trained on 3 trillion tokens

## Additional Configuration Options

The Ollama wrapper supports several additional configuration parameters:

```python
import lumen.ai as lmai

llm = lmai.llm.Ollama(
    endpoint='http://localhost:11434/v1',  # Ollama server URL
    temperature=0.7,  # Controls randomness (0-1)
    api_key='ollama',  # Required but unused
    create_kwargs={}  # Additional kwargs for API calls
)

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

## Remote Ollama Instance

You can also connect to Ollama running on a remote server:

```python
import lumen.ai as lmai

# Connect to remote Ollama instance
llm = lmai.llm.Ollama(endpoint='http://your-server:11434/v1')

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

## Tool Calling Support

Ollama now supports tool calling with popular models such as Llama 3.1. This enables models to answer prompts using available tools, making it possible to perform more complex tasks or interact with external systems.

Models that support tool calling include:
- llama3.1 (all sizes)
- llama3.2 (all sizes)
- llama3.3:70b
- qwen2.5 (7b and larger)
- qwen3 (all sizes)
- mistral-nemo
- gemma2 (9b and larger)

## Performance Considerations

- **Memory Usage**: Ollama uses sophisticated KV-cache quantization and automatic memory optimization
- **Hardware Acceleration**: Native GPU support across NVIDIA, AMD, and Apple Silicon
- **Model Size**: Choose models appropriate for your hardware - 3B models work well on most systems, while 70B+ models require significant RAM

## Troubleshooting

1. **Connection Issues**: Ensure Ollama is running with `ollama serve`
2. **Model Not Found**: Pull the required model with `ollama pull <model-name>`
3. **Performance Issues**: Consider using smaller models or enabling GPU acceleration
4. **Memory Errors**: Try quantized versions (e.g., `llama3.1:8b-q4_0`) for lower memory usage

## Streaming Support

The Ollama wrapper supports streaming responses for both text generation and structured output, providing real-time response generation for better user experience.
