#!/usr/bin/env python3
"""
Embed Vega-Lite documentation with intelligent batch processing and resume support.

This script processes Vega-Lite documentation files and creates embeddings with
contextual information using OpenAI's batch processing when beneficial.

Features:
- Automatic batch job detection and resume
- Cross-file batch processing for efficiency
- Persistent batch state management
"""

import argparse
import asyncio

from pathlib import Path

import markitdown

from tqdm import tqdm

from lumen.ai.embeddings import OpenAIEmbeddings
from lumen.ai.llm import OpenAI
from lumen.ai.vector_store import DuckDBVectorStore

# Configuration
VERSION = "6.1.0"
THIS_DIR = Path(__file__).parent
EMBEDDINGS_DIR = THIS_DIR / ".." / "lumen" / "embeddings"
BATCH_DIR = THIS_DIR / ".lumen_batch"


async def process_vega_lite_docs(
    batch_threshold: int = 1, force_new_batches: bool = False
) -> None:
    """Process Vega-Lite documentation with batch embedding.

    Args:
        batch_threshold: Minimum number of chunks for batch processing
        force_new_batches: If True, create new batches even if completed ones exist
    """
    print(f"\n🚀 Starting batch processing (threshold: {batch_threshold} chunks)")
    if force_new_batches:
        print("\n⚠️  Forcing new batches (ignoring existing completed batches)")

    # Prepare directories
    md_dir = THIS_DIR / f"vega_lite_{VERSION}_md"
    if not md_dir.exists():
        md_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n📂 Markdown directory created: {md_dir}")
        html_dir = THIS_DIR / f"vega_lite_{VERSION}"
        mid = markitdown.MarkItDown()
        for html_path in tqdm(list(html_dir.rglob("*.html"))):
            mid_response = mid.convert(html_path)
            md_dir.joinpath(html_path.relative_to(html_dir)).with_suffix(
                ".md"
            ).write_text(str(mid_response), encoding="utf-8")
    else:
        print(f"\n📂 Using existing markdown directory: {md_dir}")

    # Configure vector store with batch processing
    uri = str(EMBEDDINGS_DIR / "vega_lite.db")
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = DuckDBVectorStore(
        uri=uri,
        llm=OpenAI(),
        embeddings=OpenAIEmbeddings(),
        batch_situate_threshold=batch_threshold,
        force_new_batches=force_new_batches,
        batch_provider_kwargs={"batch_dir": str(BATCH_DIR)}
    )
    print(f"\n🔎 Vector store initialized. Connection URI: {vector_store.uri}")

    try:
        await vector_store.add_directory(
            md_dir,
            pattern="*",
            metadata={"version": VERSION, "type": "document"},
            situate=True,
        )
        print(f"\n✅ {len(vector_store)} Total chunks in vector store {uri}")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print(
            "\n💡 If this was due to a batch timeout or interruption, you can resume by running again."
        )
        raise

    finally:
        vector_store.close()


async def main(args) -> None:
    """Main entry point with argument handling."""
    await process_vega_lite_docs(
        args.batch_threshold, force_new_batches=args.force_new_batches
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Configuration arguments
    parser.add_argument(
        "--batch-threshold",
        type=int,
        default=1,
        help="Minimum number of chunks for batch processing (default: %(default)s)",
    )

    parser.add_argument(
        "--force-new-batches",
        action="store_true",
        help="Force creation of new batches even if completed ones exist",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        raise
