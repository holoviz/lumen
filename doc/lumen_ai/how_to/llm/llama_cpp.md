# Use Llama.cpp

Lumen AI supports running models in-process using the {py:class}`lumen.ai.llm.Llama` class with Llama.cpp, enabling you to leverage local models without external API calls. By default the Llama.cpp provider will fetch and use the `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` model, which strikes a good balance between hardware requirements and performance. For working with larger models we recommend using [https://llama-cpp-python.readthedocs.io/en/latest/server/] and using its [OpenAI compatible endpoints](https://llama-cpp-python.readthedocs.io/en/latest/server/).

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
lumen-ai serve <your-data-file-or-url> --provider llama
```

## Using Python

In Python, simply import the LLM wrapper {py:class}`lumen.ai.llm.Llama` and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

openai_llm = lmai.llm.Llama()

ui = lmai.ui.ExplorerUI(llm=openai_llm)
ui.servable()
```

## Configuring Models

If you do not want to use the default model (`Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`) you can override it by providing a model configuration via the `model_kwargs` parameter. This allows specifying different models for different scenarios, currently a `default` and a `reasoning` model can be provided. If no reasoning model is provided it will always use the `default` model.

As an example you can override the model configuration by providing the `repo` and `model_file` to look up on [Huggingface](https://huggingface.co/) or a `model_path` pointing to a model on disk. Any other configuration are passed through to the `llama.cpp` [`Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api) object.

As an example, let's replace Qwen 2.5 Coder with a quantized DeepSeek model we found by searching for it on [Huggingface] and then providing the repo name, model file, chat format and other configuration options:

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

llm = lmai.llm.Llama(model_kwargs=config)

lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm).servable()
```

:::{note}
Find all valid configuration options in the [Llama.cpp API reference](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#high-level-api).
:::
