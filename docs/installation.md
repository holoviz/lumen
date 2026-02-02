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

    **Get your API key:**
    
    1. Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
    2. Click "Create new secret key"
    3. Copy the key (starts with `sk-`)
    4. Set environment variable:
    
    ```bash
    # macOS/Linux
    export OPENAI_API_KEY='sk-your-key-here'
    
    # Windows PowerShell
    $env:OPENAI_API_KEY='sk-your-key-here'
    
    # Windows CMD
    set OPENAI_API_KEY=sk-your-key-here
    ```

=== "Anthropic"

    ```bash
    pip install 'lumen[ai-anthropic]'
    export ANTHROPIC_API_KEY=sk-ant-...
    ```

    **Get your API key:**
    
    1. Visit [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
    2. Click "Create Key"
    3. Copy the key (starts with `sk-ant-`)
    4. Set environment variable:
    
    ```bash
    # macOS/Linux
    export ANTHROPIC_API_KEY='sk-ant-your-key-here'
    
    # Windows PowerShell
    $env:ANTHROPIC_API_KEY='sk-ant-your-key-here'
    
    # Windows CMD
    set ANTHROPIC_API_KEY=sk-ant-your-key-here
    ```

=== "Google Gemini"

    ```bash
    pip install 'lumen[ai-google]'
    export GEMINI_API_KEY=your-key
    ```

    **Get your API key:**
    
    1. Visit [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
    2. Click "Create API key"
    3. Choose "Create API key in new project" or select existing project
    4. Copy the key (starts with `AIza`)
    5. Set environment variable:
    
    ```bash
    # macOS/Linux
    export GEMINI_API_KEY='your-key-here'
    
    # Windows PowerShell
    $env:GEMINI_API_KEY='your-key-here'
    
    # Windows CMD
    set GEMINI_API_KEY=your-key-here
    ```
    
    **Alternative:** You can also use `GOOGLE_API_KEY` instead of `GEMINI_API_KEY`.

=== "Mistral"

    ```bash
    pip install 'lumen[ai-mistralai]'
    export MISTRAL_API_KEY=your-key
    ```

    **Get your API key:**
    
    1. Visit [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys)
    2. Click "Create new key"
    3. Copy the key
    4. Set environment variable:
    
    ```bash
    # macOS/Linux
    export MISTRAL_API_KEY='your-key-here'
    
    # Windows PowerShell
    $env:MISTRAL_API_KEY='your-key-here'
    
    # Windows CMD
    set MISTRAL_API_KEY=your-key-here
    ```

=== "Azure OpenAI"

    ```bash
    pip install 'lumen[ai-openai]'
    export AZUREAI_ENDPOINT_KEY=your-key
    export AZUREAI_ENDPOINT_URL=https://your-resource.openai.azure.com/
    ```

    **Get your credentials:**
    
    1. Visit [portal.azure.com](https://portal.azure.com)
    2. Navigate to your Azure OpenAI resource
    3. Go to "Keys and Endpoint"
    4. Copy KEY 1 or KEY 2 and your endpoint URL
    5. Set environment variables:
    
    ```bash
    # macOS/Linux
    export AZUREAI_ENDPOINT_KEY='your-key-here'
    export AZUREAI_ENDPOINT_URL='https://your-resource.openai.azure.com/'
    
    # Windows PowerShell
    $env:AZUREAI_ENDPOINT_KEY='your-key-here'
    $env:AZUREAI_ENDPOINT_URL='https://your-resource.openai.azure.com/'
    
    # Windows CMD
    set AZUREAI_ENDPOINT_KEY=your-key-here
    set AZUREAI_ENDPOINT_URL=https://your-resource.openai.azure.com/
    ```

=== "Azure Mistral AI"

    ```bash
    pip install 'lumen[ai-mistralai]'
    export AZUREAI_ENDPOINT_KEY=your-key
    export AZUREAI_ENDPOINT_URL=https://your-resource-endpoint.com/
    ```

    **Get your credentials:**
    
    1. Visit [portal.azure.com](https://portal.azure.com)
    2. Navigate to your Azure AI resource with Mistral deployment
    3. Go to "Keys and Endpoint"
    4. Copy KEY 1 or KEY 2 and your endpoint URL
    5. Set environment variables (same as Azure OpenAI above)

---

## Locally Hosted

Run open-source LLMs on your own machine. No API keys required, full privacy, free to use.

=== "Ollama"

    ```bash
    pip install 'lumen[ai-ollama]'
    ```

    **Setup Ollama:**
    
    1. Install Ollama from [ollama.com](https://ollama.com)
    2. Start the Ollama service (usually starts automatically)
    3. Pull a model:
    
    ```bash
    ollama pull qwen3:32b
    ```
    
    4. (Optional) Set custom endpoint if not using default:
    
    ```bash
    export OLLAMA_BASE_URL=http://localhost:11434
    ```
    
    **No additional environment variables needed!** Ollama works out of the box.

=== "Llama.cpp"

    ```bash
    pip install 'lumen[ai-llama]'
    ```

    **No setup required!** The first time you use Llama.cpp, Lumen will automatically download the model you specify. No environment variables needed.
    
    **Optional:** Set a custom model URL:
    
    ```bash
    lumen-ai serve data.csv --provider llama-cpp \
      --llm-model-url 'https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf'
    ```

=== "AI Navigator"

    ```bash
    pip install 'lumen[ai-openai]'
    ```

    [Anaconda AI Navigator](https://www.anaconda.com/products/ai-navigator) runs models locally on your machine. No API key needed!
    
    **Default endpoint:** `http://localhost:8080/v1`
    
    **Optional:** Set custom endpoint:
    
    ```bash
    export AINAVIGATOR_BASE_URL=http://localhost:8080/v1
    ```

=== "AI Catalyst"

    ```bash
    pip install 'lumen[ai-openai]'
    export AI_CATALYST_BASE_URL=https://...
    export AI_CATALYST_API_KEY=...
    ```

    [Anaconda AI Catalyst](https://www.anaconda.com/platform/ai-catalyst) provides enterprise-ready, governed models running on your private infrastructure.

    **Get your credentials:**
    
    1. Log in to your AI Catalyst dashboard
    2. Go to **Model Servers**, launch or select a server, and copy the **Server Address**
    3. Go to your profile icon > **API Keys** and create/copy a key
    4. Set environment variables:

    ```bash
    # macOS/Linux
    export AI_CATALYST_BASE_URL='https://your-company.anacondaconnect.com/.../v1'
    export AI_CATALYST_API_KEY='your-key-here'
    
    # Windows PowerShell
    $env:AI_CATALYST_BASE_URL='https://your-company.anacondaconnect.com/.../v1'
    $env:AI_CATALYST_API_KEY='your-key-here'
    ```

---

## Router / Multi-Provider

Use a unified interface to access multiple LLM providers and models.

=== "LiteLLM"

    ```bash
    pip install 'lumen[ai-litellm]'
    ```

    Supports 100+ models across OpenAI, Anthropic, Google, Mistral, and more.

    **Set environment variables for providers you want to use:**

    === "OpenAI"

        ```bash
        export OPENAI_API_KEY='sk-...'
        ```

    === "Anthropic"

        ```bash
        export ANTHROPIC_API_KEY='sk-ant-...'
        ```

    === "Google"

        ```bash
        export GEMINI_API_KEY='...'
        ```

    === "Azure"

        ```bash
        export AZUREAI_ENDPOINT_KEY='...'
        export AZUREAI_ENDPOINT_URL='https://...'
        ```

    === "AWS Bedrock"

        === "aws-sso-util (Easiest)"
        
            ```bash
            pip install aws-sso-util
            aws-sso-util login --profile your-profile
            # Credentials auto-exported!
            ```
        
        === "AWS CLI"
        
            ```bash
            aws sso login --profile your-profile
            export AWS_PROFILE=your-profile
            export AWS_DEFAULT_REGION=us-east-1
            ```
        
        === "Access Keys"
        
            ```bash
            export AWS_ACCESS_KEY_ID='...'
            export AWS_SECRET_ACCESS_KEY='...'
            export AWS_DEFAULT_REGION='us-east-1'
            ```

    Then use any supported model:

    ```python
    import lumen.ai as lmai

    llm = lmai.llm.LiteLLM(
        model_kwargs={
            "default": {"model": "gpt-4.1-mini"},                    # OpenAI
            "edit": {"model": "anthropic/claude-sonnet-4-5"},        # Anthropic
            "sql": {"model": "gemini/gemini-2.5-flash"}             # Google
        }
    )
    ui = lmai.ExplorerUI(data='data.csv', llm=llm)
    ui.servable()
    ```

=== "AWS Bedrock"

    AWS Bedrock is a managed gateway that provides access to foundation models from Anthropic, Meta, Mistral, Amazon, Cohere, and AI21 through a unified API.

    ```bash
    pip install 'lumen[ai-anthropic]'  # For AnthropicBedrock
    # OR
    pip install boto3  # For Bedrock
    ```

    **Authentication:**

    === "aws-sso-util (Easiest)"
    
        ```bash
        pip install aws-sso-util
        aws-sso-util login --profile your-profile
        # Credentials auto-exported!
        ```
    
    === "AWS CLI"
    
        ```bash
        aws sso login --profile your-profile
        export AWS_PROFILE=your-profile
        export AWS_DEFAULT_REGION=us-east-1
        ```
    
    === "Access Keys"
    
        ```bash
        export AWS_ACCESS_KEY_ID='...'
        export AWS_SECRET_ACCESS_KEY='...'
        export AWS_DEFAULT_REGION='us-east-1'
        ```

    **Choose your Lumen provider:**

    === "AnthropicBedrock"

        Optimized for Claude models using Anthropic's SDK.

        ```python
        import lumen.ai as lmai

        llm = lmai.llm.AnthropicBedrock()
        ui = lmai.ExplorerUI(data='data.csv', llm=llm)
        ui.servable()
        ```

    === "Bedrock"

        Universal access to all Bedrock models using boto3.

        ```python
        import lumen.ai as lmai
        
        llm = lmai.llm.Bedrock(
            model_kwargs={
                "default": {"model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
            }
        )
        ui = lmai.ExplorerUI(data='data.csv', llm=llm)
        ui.servable()
        ```

        **Available models:**
        
        - Anthropic (Claude), Meta (Llama), Mistral, Amazon (Titan), Cohere, AI21
        - Model IDs: `us.anthropic.claude-*`, `meta.llama3-*`, `mistral.*`, `amazon.titan-*`
        - [Full model list](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)

    **IAM Permissions:**

    ```json
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "*"
    }
    ```

---

## Making Environment Variables Persistent

Set variables permanently so you don't have to export them every session:

=== "macOS/Linux (Bash)"

    Add to `~/.bashrc` or `~/.bash_profile`:
    
    ```bash
    echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
    source ~/.bashrc
    ```

=== "macOS (Zsh)"

    Add to `~/.zshrc`:
    
    ```bash
    echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.zshrc
    source ~/.zshrc
    ```

=== "Windows (Permanent)"

    1. Search for "Environment Variables" in Start Menu
    2. Click "Edit the system environment variables"
    3. Click "Environment Variables"
    4. Under "User variables", click "New"
    5. Add variable name (e.g., `OPENAI_API_KEY`) and value
    6. Click OK
    7. Restart your terminal

---

## Verify Installation

```bash
lumen-ai --version
```

Test your LLM connection:

```python
# Test script
import lumen.ai as lmai

llm = lmai.llm.OpenAI()  # or Anthropic(), Google(), etc.
ui = lmai.ExplorerUI(data='test.csv')
ui.servable()
```

---

## Next Steps

Ready to start? Head to the [Quick Start](quick_start.md) guide to chat with your first dataset.

---

## Missing Your Favorite LLM?

Missing your favorite LLM? Let us know by submitting a [GitHub issue](https://github.com/holoviz/lumen/issues)!
