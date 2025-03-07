# Use Llama.cpp

Lumen AI supports running models in-process using the {py:class}`lumen.ai.llm.LlamaCpp` class with Llama.cpp, enabling you to leverage local models without external API calls. By default the Llama.cpp provider will fetch and use the `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` model, which strikes a good balance between hardware requirements and performance. For working with larger models we recommend using [https://llama-cpp-python.readthedocs.io/en/latest/server/] and using its [OpenAI compatible endpoints](https://llama-cpp-python.readthedocs.io/en/latest/server/).

:::{note}
When using the Llama.cpp provider the first time it will download the specified model, which may take some time.
:::

## Prerequisites

- Lumen AI installed in your Python environment.
- Llama.cpp installed and configured on your system. [Follow the Llama.cpp Installation Guide](https://llama-cpp-python.readthedocs.io/en/latest/).
- Decent hardware such as a modern GPU, an ARM based Mac or a high-core count CPU.

## Using CLI Arguments

Once configured you can select llama.cpp as the provider using a CLI argument:

```bash
lumen-ai serve <your-data-file-or-url> --provider llama-cpp
```

## Using Python

In Python, simply import the LLM wrapper {py:class}`lumen.ai.llm.LlamaCpp` and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

openai_llm = lmai.llm.LlamaCpp()

ui = lmai.ui.ExplorerUI(llm=openai_llm)
ui.servable()
```

## Configuring Models

If you do not want to use the default model (`Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`) you can override it by providing a model configuration via the `model_kwargs` parameter. This allows specifying different models for different scenarios, currently a `default` and a `reasoning` model can be provided. If no reasoning model is provided it will always use the `default` model.

As an example you can override the model configuration by providing the `repo` and `model_file` to look up on [Huggingface](https://huggingface.co/) or a `model_path` pointing to a model on disk. Any other configuration are passed through to the `llama.cpp` [`LlamaCpp`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api) object.

As an example, let's replace Qwen 2.5 Coder with a quantized DeepSeek model we found by searching for it on [Huggingface] and then providing the repo name, model file, chat format and other configuration options in Python:

```python
import lumen.ai as lmai

config = {
    "default": {
        "repo": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "model_file": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "chat_format": "qwen",
        "n_ctx": 131072,
    }
}

llm = lmai.llm.LlamaCpp(model_kwargs=config)

lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm).servable()
```

:::{note}
Find all valid configuration options in the [Llama.cpp API reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api).
:::

Using another model can be done in the CLI as well:

```bash
lumen-ai serve --provider llama --model-kwargs '{
  "default": {
    "repo": "unsloth/Mistral-Small-24B-Instruct-2501-GGUF",
    "model_file": "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
    "chat_format": "mistral-instruct"
  }
}'
```

Providing these arguments via the CLI can be cumbersome. Instead, paste the quantized model file's `llm-model-url` and pass `model_kwargs` as query parameters. If `llm-model-url` is set, `provider` automatically defaults to `llama`, and will error if another `provider` is set.

```bash
lumen-ai serve --llm-model-url https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF/blob/main/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf?chat_format=mistral-instruct
```

```bash
lumen-ai serve --llm-model-url https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf?chat_format=qwen&n_ctx=131072
```
