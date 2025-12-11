# :material-vector-polygon: Embeddings

Convert text into numerical vectors for semantic search.

## Quick start

By default, Lumen uses simple Numpy-based embeddings:

``` py title="Default embeddings (no setup needed)"
import lumen.ai as lmai

ui = lmai.ExplorerUI(data='penguins.csv')
ui.servable()
```

For better semantic search, use OpenAI:

``` py title="OpenAI embeddings" hl_lines="3-4"
import lumen.ai as lmai
from lumen.ai.embeddings import OpenAIEmbeddings
from lumen.ai.vector_store import DuckDBVectorStore

vector_store = DuckDBVectorStore(embeddings=OpenAIEmbeddings())
ui = lmai.ExplorerUI(data='penguins.csv', vector_store=vector_store)
ui.servable()
```

See [Vector Stores](vector_stores.md) for how to use embeddings with storage backends.

## Providers

### OpenAI

``` py title="OpenAI embeddings"
from lumen.ai.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Fast
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Higher quality
```

**Setup:**

``` bash
export OPENAI_API_KEY="sk-..."
```

See [LLM Providers](llm_providers.md) for more on OpenAI configuration.

### Azure OpenAI

``` py title="Azure embeddings"
from lumen.ai.embeddings import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    api_key='...',
    endpoint='https://your-resource.openai.azure.com/'
)
```

See [LLM Providers - Azure OpenAI](llm_providers.md#azure-openai) for authentication details.

### HuggingFace

Run locally with Sentence Transformers:

``` py title="Local embeddings"
from lumen.ai.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model="ibm-granite/granite-embedding-107m-multilingual",
    device="cpu"  # or "cuda" for GPU
)
```

**Install:**

``` bash
pip install sentence-transformers
```

### Llama.cpp

Run GGUF models locally:

``` py title="GGUF embeddings"
from lumen.ai.embeddings import LlamaCppEmbeddings

embeddings = LlamaCppEmbeddings(
    model_kwargs={
        "default": {
            "repo_id": "Qwen/Qwen3-Embedding-4B-GGUF",
            "filename": "Qwen3-Embedding-4B-Q4_K_M.gguf",
        }
    }
)
```

See [LLM Providers - Llama.cpp](llm_providers.md#llamacpp-local) for model configuration.

### Numpy (default)

Hash-based embeddings for prototyping:

``` py title="Simple embeddings"
from lumen.ai.embeddings import NumpyEmbeddings

embeddings = NumpyEmbeddings()
```

- ✅ No API calls, works offline
- ⚠️ Lower quality than neural embeddings

## Configuration

### Chunk size

Control how documents are split:

``` py title="Custom chunking" hl_lines="3"
vector_store = DuckDBVectorStore(
    embeddings=OpenAIEmbeddings(),
    chunk_size=512,  # Smaller chunks = more precise
)
```

**Guidelines:**

- Small (256-512): Precise answers, higher cost
- Medium (1024): Balanced (default)
- Large (2048): Broader context, lower cost

See [Vector Stores - Chunk size](vector_stores.md#chunk-size) for implementation details.

### Exclude metadata

Prevent fields from being embedded:

``` py title="Exclude metadata" hl_lines="3"
vector_store = DuckDBVectorStore(
    embeddings=OpenAIEmbeddings(),
    excluded_metadata=['file_size', 'upload_date']
)
```

## When embeddings are used

Embeddings power three features:

**Document search** - Queries find semantically similar text:

``` py
results = await vector_store.query('authentication setup')
```

See [Vector Stores - Searching](vector_stores.md#searching) for query examples.

**Table discovery** - Tools find relevant tables:

``` py
from lumen.ai.tools import IterativeTableLookup

tool = IterativeTableLookup(tables=['customers', 'orders', 'products'])
```

See [Tools - Built-in tools](tools.md#built-in-tools) for tool configuration.

**Contextual augmentation** - Chunks get context descriptions:

``` py
vector_store = DuckDBVectorStore(situate=True)  # Adds context to chunks
```

See [Vector Stores - Contextual augmentation](vector_stores.md#contextual-augmentation-situate) for details on situate.

## Best practices

**Match embeddings to data:**

- English-only → `sentence-transformers/all-MiniLM-L6-v2`
- Multilingual → `ibm-granite/granite-embedding-107m-multilingual`
- Best quality → `text-embedding-3-large`

**Optimize chunk size:**

- FAQ/short answers → 256-512 tokens
- General documents → 1024 tokens (default)
- Long-form content → 2048 tokens

**Use situate selectively:**

- Enable for technical docs, books, research papers
- Disable for simple content (FAQs, short articles)
- Requires LLM access (uses additional API calls)

## See also

- [Vector Stores](vector_stores.md) - Storage and retrieval using embeddings
- [LLM Providers](llm_providers.md) - Configure API keys and models
- [Tools](tools.md) - Built-in tools that use embeddings
