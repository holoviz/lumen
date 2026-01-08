# :material-database-search: Vector Stores

Store and search text using semantic similarity.

## Quick start

In-memory vector store:

``` py title="Simple vector store"
from lumen.ai.vector_store import NumpyVectorStore

vector_store = NumpyVectorStore()
await vector_store.add_file('documentation.pdf')

results = await vector_store.query('authentication setup', top_k=3)
```

Persistent storage with DuckDB:

``` py title="Persistent vector store"
from lumen.ai.vector_store import DuckDBVectorStore

vector_store = DuckDBVectorStore(uri='embeddings.db')
await vector_store.add_file('documentation.pdf')
```

See [Embeddings](embeddings.md) for configuring how text is converted to vectors.

## Store types

### NumpyVectorStore

In-memory storage using Numpy arrays.

``` py
from lumen.ai.vector_store import NumpyVectorStore

vector_store = NumpyVectorStore()
```

- ✅ Fast, simple
- ⚠️ Data lost on restart
- Best for: Development, testing, small datasets

### DuckDBVectorStore

Persistent storage with HNSW indexing.

``` py
from lumen.ai.vector_store import DuckDBVectorStore

vector_store = DuckDBVectorStore(uri='embeddings.db')
```

- ✅ Persists on disk
- ✅ Scales to millions of documents
- ✅ Fast similarity search
- Best for: Production, large datasets

## Adding documents

### Add files

``` py title="Add any file"
await vector_store.add_file('documentation.pdf')
await vector_store.add_file('guide.md')
await vector_store.add_file('https://example.com/page')  # URLs work too
```

### Add directories

``` py title="Add all files"
await vector_store.add_directory(
    'docs/',
    pattern='*.md',                  # Only markdown
    exclude_patterns=['**/draft/*'], # Skip drafts
    max_concurrent=10                # Process 10 at once
)
```

### Add text

``` py title="Add text directly"
await vector_store.add([
    {
        'text': 'Lumen is a data exploration framework.',
        'metadata': {'source': 'intro', 'category': 'overview'}
    }
])
```

## Searching

### Semantic search

``` py title="Find similar text"
results = await vector_store.query(
    'How do I authenticate users?',
    top_k=5,        # Top 5 results
    threshold=0.3   # Min similarity
)

for result in results:
    print(f"{result['similarity']:.2f}: {result['text']}")
```

Similarity is powered by embeddings - see [Embeddings - Providers](embeddings.md#providers) for quality options.

### Filter by metadata

``` py title="Filter search"
results = await vector_store.query(
    'authentication',
    filters={'category': 'security', 'version': '2.0'}
)
```

### Exact metadata lookup

``` py title="Metadata-only search"
results = vector_store.filter_by(
    filters={'author': 'admin'},
    limit=10
)
```

## Upsert (prevent duplicates)

``` py title="Upsert instead of add"
# First call - adds new item
await vector_store.upsert([
    {'text': 'Hello world', 'metadata': {'source': 'greeting'}}
])

# Second call - skips (already exists)
await vector_store.upsert([
    {'text': 'Hello world', 'metadata': {'source': 'greeting'}}
])
```

Use `upsert()` when reprocessing documents that may not have changed.

## Management

``` py title="Delete, clear, count"
# Delete by ID
vector_store.delete([1, 2, 3])

# Clear everything
vector_store.clear()

# Count documents
num_docs = len(vector_store)
```

## Contextual augmentation (situate)

Add context descriptions to chunks:

``` py title="Enable situate" hl_lines="2"
vector_store = DuckDBVectorStore(
    situate=True,  # Generate context for each chunk
)

await vector_store.add_file('long_document.pdf')
```

Each chunk gets context like:

> "This section discusses OAuth2 authentication. It follows the introduction and references token refresh mechanisms."

**When to use:**

- ✅ Long technical documents, books, research papers
- ✅ Documents with forward/backward references
- ❌ Short documents, FAQs, independent chunks

Requires an LLM to generate context - see [LLM Providers](llm_providers.md) for configuration.

## Integration with Lumen AI

### Document search

``` py title="Enable document search"
import lumen.ai as lmai

vector_store = DuckDBVectorStore(uri='docs.db')
await vector_store.add_directory('documentation/')

ui = lmai.ExplorerUI(
    data='penguins.csv',
    vector_store=vector_store
)
ui.servable()
```

Users can now ask questions about uploaded documents. See [Tools - DocumentLookup](tools.md#built-in-tools) for how this works.

## Configuration

### Custom embeddings

``` py title="Use different embeddings"
from lumen.ai.embeddings import HuggingFaceEmbeddings

vector_store = DuckDBVectorStore(
    embeddings=HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
)
```

See [Embeddings - Providers](embeddings.md#providers) for all embedding options.

### Read-only mode

``` py title="Read-only access"
vector_store = DuckDBVectorStore(
    uri='embeddings.db',
    read_only=True
)
```

### Chunk size

``` py title="Control chunking"
vector_store = DuckDBVectorStore(
    chunk_size=512  # Smaller chunks
)
```

See [Embeddings - Chunk size](embeddings.md#chunk-size) for chunking strategies.

## Best practices

**Choose the right store:**

- Development → `NumpyVectorStore`
- Production → `DuckDBVectorStore`

**Optimize threshold:**

- Exploratory → `threshold=0.3`
- Precise → `threshold=0.5`
- Very strict → `threshold=0.7`

**Use upsert for idempotency:**

- Reprocessing → `upsert()`
- New content → `add()`

## See also

- [Embeddings](embeddings.md) - Configure how text is converted to vectors
- [Tools](tools.md) - Built-in tools that use vector stores
- [Agents](agents.md) - Agents that leverage document search
- [LLM Providers](llm_providers.md) - Required for situate feature
