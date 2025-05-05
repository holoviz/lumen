import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.embeddings import NumpyEmbeddings
from lumen.ai.vector_store import DuckDBVectorStore, NumpyVectorStore


@pytest.mark.xdist_group("vss")
class VectorStoreTestKit:
    """
    A base class (test kit) that provides the *common* tests and fixture definitions.
    """

    @pytest.fixture
    def store(self):
        """
        Must be overridden in the subclass to return a fresh store instance.
        """
        raise NotImplementedError("Subclasses must override `store` fixture")

    @pytest.fixture
    def empty_store(self, store):
        """
        Returns an empty store (clears whatever the `store` fixture gave us).
        """
        store.clear()
        return store

    @pytest.fixture
    async def store_with_three_docs(self, store):
        """
        Returns a store preloaded with three documents for convenience.
        """
        items = [
            {
                "text": "Food: $10, Drinks: $5, Total: $15",
                "metadata": {"title": "receipt", "department": "accounting"},
            },
            {
                "text": "In the org chart, the CEO reports to the board.",
                "metadata": {"title": "org_chart", "department": "management"},
            },
            {
                "text": "A second receipt with different details",
                "metadata": {"title": "receipt", "department": "accounting"},
            },
        ]
        await store.add(items)
        return store

    @pytest.mark.asyncio
    async def test_filter_by_title_receipt(self, store_with_three_docs):
        filtered = store_with_three_docs.filter_by({"title": "receipt"})
        assert len(filtered) == 2

    @pytest.mark.asyncio
    async def test_filter_by_department_management(self, store_with_three_docs):
        filtered_mgmt = store_with_three_docs.filter_by({"department": "management"})
        assert len(filtered_mgmt) == 1
        assert filtered_mgmt[0]["metadata"]["title"] == "org_chart"

    @pytest.mark.asyncio
    async def test_filter_by_nonexistent_filter(self, store_with_three_docs):
        filtered_none = store_with_three_docs.filter_by({"title": "does_not_exist"})
        assert len(filtered_none) == 0

    @pytest.mark.asyncio
    async def test_filter_by_limit_and_offset(self, store_with_three_docs):
        filtered_limited = store_with_three_docs.filter_by({"title": "receipt"}, limit=1)
        assert len(filtered_limited) == 1

        filtered_offset = store_with_three_docs.filter_by({"title": "receipt"}, limit=1, offset=1)
        assert len(filtered_offset) == 1
        assert filtered_offset[0]["text"] == "A second receipt with different details"

    @pytest.mark.asyncio
    async def test_query_with_filter_title_org_chart(self, store_with_three_docs):
        query_text = "CEO reports to?"
        results = store_with_three_docs.query(query_text)
        assert len(results) >= 1
        assert results[0]["metadata"]["title"] == "org_chart"
        assert "CEO" in results[0]["text"]

        results = store_with_three_docs.query(query_text, filters={"title": "org_chart"})
        assert len(results) == 1
        assert results[0]["metadata"]["title"] == "org_chart"
        assert "CEO" in results[0]["text"]

    @pytest.mark.asyncio
    async def test_query_1_threshold(self, store_with_three_docs):
        text = "Food: $10, Drinks: $5, Total: $15"
        metadata = {"title": "receipt", "department": "accounting"}
        stored_embedding_text = store_with_three_docs._join_text_and_metadata(text, metadata)
        results = store_with_three_docs.query(stored_embedding_text, threshold=0.99)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_empty_store(self, empty_store):
        results = empty_store.query("some query")
        assert results == []

    @pytest.mark.asyncio
    async def test_filter_empty_store(self, empty_store):
        filtered = empty_store.filter_by({"key": "value"})
        assert filtered == []

    @pytest.mark.asyncio
    async def test_delete_empty_store(self, empty_store):
        empty_store.delete([1, 2, 3])
        results = empty_store.query("some query")
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, store_with_three_docs):
        original_count = len(store_with_three_docs.filter_by({}))
        store_with_three_docs.delete([])
        after_count = len(store_with_three_docs.filter_by({}))
        assert original_count == after_count

    @pytest.mark.asyncio
    async def test_delete_specific_ids(self, store_with_three_docs):
        all_docs = store_with_three_docs.filter_by({})
        target_id = all_docs[0]["id"]
        store_with_three_docs.delete([target_id])

        remaining_docs = store_with_three_docs.filter_by({})
        remaining_ids = [doc["id"] for doc in remaining_docs]
        assert target_id not in remaining_ids

    @pytest.mark.asyncio
    async def test_clear_store(self, store_with_three_docs):
        store_with_three_docs.clear()
        results = store_with_three_docs.filter_by({})
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_add_docs_without_metadata(self, empty_store):
        items = [
            {"text": "Document one with no metadata"},
            {"text": "Document two with no metadata"},
        ]
        ids = await empty_store.add(items)
        assert len(ids) == 2

        results = empty_store.query("Document one")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_add_docs_with_nonstring_metadata(self, empty_store):
        items = [
            {
                "text": "Doc with int metadata",
                "metadata": {"count": 42, "description": None},
            },
            {
                "text": "Doc with list metadata",
                "metadata": {"tags": ["foo", "bar"]},
            },
        ]
        await empty_store.add(items)

        results = empty_store.filter_by({})
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_add_long_text_chunking(self, empty_store):
        """
        Verifies that adding a document with text longer than `chunk_size` is chunked properly.
        """
        # For demonstration, set a small chunk_size
        empty_store.chunk_size = 50  # force chunking fairly quickly

        # Create a long text, repeated enough times to exceed the chunk_size
        long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
        items = [
            {
                "text": long_text,
                "metadata": {"title": "long_document"},
            }
        ]
        ids = await empty_store.add(items)

        # Should have multiple chunks, hence multiple IDs
        # (The exact number depends on chunk_size & text length.)
        assert len(ids) > 1, "Expected more than one chunk/ID due to forced chunking"

    @pytest.mark.asyncio
    async def test_query_long_text_chunking(self, empty_store):
        """
        Verifies querying a store containing a large text still returns sensible results.
        """
        empty_store.chunk_size = 60
        long_text = " ".join(["word"] * 300)  # 300 words, ensures multiple chunks

        items = [
            {
                "text": long_text,
                "metadata": {"title": "very_large_document"},
            }
        ]
        await empty_store.add(items)

        # Query for a word we know is in the text.
        # Not a perfect test since this is a mocked embedding, but ensures no errors.
        results = empty_store.query("word", top_k=3)
        assert len(results) <= 3, "We requested top_k=3"
        assert len(results) > 0, "Should have at least one chunk matching"

        # The top result should logically be from our big text.
        top_result = results[0]
        assert "very_large_document" in str(top_result.get("metadata")), (
            "Expected the big doc chunk to show up in the results"
        )

    @pytest.mark.asyncio
    async def test_add_multiple_large_documents(self, empty_store):
        """
        Verifies behavior when multiple large documents are added.
        """
        # Force chunking to a smaller chunk_size
        empty_store.chunk_size = 100

        doc1 = "Doc1 " + ("ABC " * 200)
        doc2 = "Doc2 " + ("XYZ " * 200)
        items = [
            {
                "text": doc1,
                "metadata": {"title": "large_document_1"},
            },
            {
                "text": doc2,
                "metadata": {"title": "large_document_2"},
            },
        ]

        ids = await empty_store.add(items)
        # At least more than 2 chunks, since each doc is forced to chunk
        assert len(ids) > 2, "Expected more than 2 chunks total for two large docs"

        # Query something from doc2
        results = empty_store.query("XYZ", top_k=5)
        assert len(results) <= 5
        # Expect at least 1 chunk from doc2
        # This is a simplistic check, but ensures chunking doesn't break queries
        found_doc2 = any("large_document_2" in r["metadata"].get("title", "") for r in results)
        assert found_doc2, "Expected to find at least one chunk belonging to doc2"

        # Query something from doc1
        results = empty_store.query("ABC", top_k=5)
        assert len(results) <= 5
        found_doc1 = any("large_document_1" in r["metadata"].get("title", "") for r in results)
        assert found_doc1, "Expected to find at least one chunk belonging to doc1"

    @pytest.mark.asyncio
    async def test_upsert_new_item(self, empty_store):
        """
        Verifies that upsert adds a new item when it doesn't exist.
        """
        item = {"text": "A new document", "metadata": {"source": "test"}}
        ids = await empty_store.upsert([item])
        assert len(ids) == 1, "Should add one item"

        results = empty_store.query("A new document")
        assert len(results) == 1, "Should be able to query the added item"
        assert results[0]["metadata"]["source"] == "test", "Metadata should match"

    @pytest.mark.asyncio
    async def test_upsert_skip_duplicate(self, empty_store):
        """
        Verifies that upsert skips items with high similarity and matching metadata.
        """
        # Add initial item
        item1 = {"text": "Hello world!", "metadata": {"source": "greeting"}}
        ids1 = await empty_store.upsert([item1])
        assert len(ids1) == 1, "Should add one item"

        # Try to add same item again - should be skipped
        item2 = {"text": "Hello world!", "metadata": {"source": "greeting"}}
        ids2 = await empty_store.upsert([item2])
        assert len(ids2) == 1, "Should return one ID"
        assert ids2[0] == ids1[0], "Should return the same ID as before"

        # Check store size
        all_docs = empty_store.filter_by({})
        assert len(all_docs) == 1, "Should still have only one item"

    @pytest.mark.asyncio
    async def test_upsert_add_with_different_metadata(self, empty_store):
        """
        Verifies that upsert adds items with similar text but different metadata.
        """
        # Add initial item
        item1 = {"text": "Hello universe!", "metadata": {"source": "greeting1"}}
        ids1 = await empty_store.upsert([item1])
        assert len(ids1) == 1, "Should add one item"

        # Add similar item but with different metadata
        item2 = {"text": "Hello universe!", "metadata": {"source": "greeting2"}}
        ids2 = await empty_store.upsert([item2])
        assert len(ids2) == 1, "Should add one item"
        assert ids2[0] != ids1[0], "Should have a different ID"

        # Check store size
        all_docs = empty_store.filter_by({})
        assert len(all_docs) == 2, "Should have two items"

    @pytest.mark.asyncio
    async def test_upsert_multiple_items(self, empty_store):
        """
        Verifies that upsert handles multiple items correctly.
        """
        # Add initial items
        items1 = [
            {"text": "Item one", "metadata": {"id": 1}},
            {"text": "Item two", "metadata": {"id": 2}},
        ]
        ids1 = await empty_store.upsert(items1)
        assert len(ids1) == 2, "Should add two items"

        # Try to add a mix of duplicate and new items
        items2 = [
            {"text": "Item one", "metadata": {"id": 1}},  # Duplicate
            {"text": "Item three", "metadata": {"id": 3}},  # New
        ]
        ids2 = await empty_store.upsert(items2)
        assert len(ids2) == 2, "Should return two IDs"
        assert ids2[0] == ids1[0], "First ID should be the same as before"
        assert ids2[1] != ids1[1], "Second ID should be different"

        # Check store size
        all_docs = empty_store.filter_by({})
        assert len(all_docs) == 3, "Should have three items total"

    @pytest.mark.asyncio
    async def test_upsert_with_additional_metadata(self, empty_store):
        """
        Verifies that upsert correctly handles adding items with identical text
        but with additional metadata fields.
        """
        # Add initial item with minimal metadata
        item1 = {"text": "Hello Python world!", "metadata": {"language": "python"}}
        ids1 = await empty_store.upsert([item1])
        assert len(ids1) == 1, "Should add one item"

        # Query to verify initial content
        results1 = empty_store.query("Hello Python world!")
        assert len(results1) == 1, "Should find the item"
        assert results1[0]["text"] == "Hello Python world!"

        # Now upsert the same text with additional metadata fields
        item2 = {
            "text": "Hello Python world!",
            "metadata": {
                "language": "python",  # Same as before
                "version": "3.9",      # Additional field
                "purpose": "test"      # Additional field
            }
        }
        ids2 = await empty_store.upsert([item2])
        assert len(ids2) == 1, "Should return one ID"

        # Should be treated as an update, not a new item
        all_docs = empty_store.filter_by({})
        assert len(all_docs) == 1, "Should still have only one item"

        # Check that metadata was updated
        updated_item = empty_store.filter_by({"language": "python"})[0]
        assert "version" in updated_item["metadata"], "Should have updated with new metadata fields"
        assert updated_item["metadata"]["version"] == "3.9", "New metadata value should be present"

    @pytest.mark.asyncio
    async def test_upsert_with_exact_same_content(self, empty_store):
        """
        Verifies that upsert doesn't create duplicates when adding the exact same
        text and metadata multiple times.
        """
        # Add initial item
        item = {"text": "print('Hello!')", "metadata": {"language": "py"}}

        # Upsert multiple times with identical content
        ids1 = await empty_store.upsert([item])
        ids2 = await empty_store.upsert([item])
        ids3 = await empty_store.upsert([item])

        assert len(ids1) == 1, "Should add one item on first upsert"
        assert len(ids2) == 1, "Should return one ID on second upsert"
        assert len(ids3) == 1, "Should return one ID on third upsert"

        # All returned IDs should be identical
        assert ids1[0] == ids2[0], "First and second upsert should return the same ID"
        assert ids2[0] == ids3[0], "Second and third upsert should return the same ID"

        # Should only have one item in the store
        all_docs = empty_store.filter_by({})
        assert len(all_docs) == 1, "Should have only one item after multiple upserts of same content"

    @pytest.mark.asyncio
    async def test_upsert_with_removed_metadata(self, empty_store):
        """
        Verifies that upsert correctly handles removing metadata keys.
        """
        # Add initial item with multiple metadata fields
        item1 = {
            "text": "Metadata testing example",
            "metadata": {
                "key1": "value1",
                "key2": "value2",
                "key3": "value3"
            }
        }
        ids1 = await empty_store.upsert([item1])
        assert len(ids1) == 1, "Should add one item"

        # Verify initial state
        results = empty_store.filter_by({"key1": "value1"})
        assert len(results) == 1, "Should find the item"
        assert len(results[0]["metadata"]) == 3, "Should have three metadata keys"
        assert "key3" in results[0]["metadata"], "key3 should be present"

        # Now upsert with fewer metadata keys
        item2 = {
            "text": "Metadata testing example",
            "metadata": {
                "key1": "value1",
                "key2": "value2"
                # key3 is removed
            }
        }
        ids2 = await empty_store.upsert([item2])
        assert len(ids2) == 1, "Should return one ID"
        assert ids2[0] == ids1[0], "Should return the same ID as before"

        # Verify metadata was updated (key3 was removed)
        results = empty_store.filter_by({"key1": "value1"})
        assert len(results) == 1, "Should still have only one item"
        assert len(results[0]["metadata"]) == 2, "Should now have only two metadata keys"
        assert "key3" not in results[0]["metadata"], "key3 should be removed"

    def test_upsert_empty_list(self, empty_store):
        """
        Verifies that upsert handles empty input gracefully.
        """
        ids = empty_store.upsert([])
        assert ids == [], "Should return empty list of IDs"

        # Store should remain empty
        all_docs = empty_store.filter_by({})
        assert len(all_docs) == 0, "Store should still be empty"

    def test_upsert_long_content_no_duplication(self, empty_store):
        """
        Verifies that upsert doesn't create duplicates when adding the same long text
        that gets chunked.
        """
        # Set a small chunk size to force chunking
        empty_store.chunk_size = 100

        # Create a long text that will be split into multiple chunks
        long_text = "Test document that is long enough to be split into multiple chunks. " * 10
        metadata = {"source": "test", "type": "long_document"}

        # First upsert
        item = {"text": long_text, "metadata": metadata}
        ids1 = empty_store.upsert([item])

        # Verify chunking occurred
        assert len(ids1) > 1, "Text should be split into multiple chunks"

        # Record count after first upsert
        count_after_first = len(empty_store)

        # Perform second upsert with the same content
        empty_store.upsert([item])
        count_after_second = len(empty_store)

        # Verify no new entries were created
        assert count_after_first == count_after_second, "No new items should be added on second upsert"


class TestNumpyVectorStore(VectorStoreTestKit):

    @pytest.fixture
    def store(self):
        store = NumpyVectorStore(embeddings=NumpyEmbeddings())
        store.clear()
        return store


class TestDuckDBVectorStore(VectorStoreTestKit):

    @pytest.fixture
    def store(self, tmp_path) -> DuckDBVectorStore:
        db_path = str(tmp_path / "test_duckdb.db")
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        store.clear()
        return store

    def test_persistence(self, tmp_path):
        db_path = str(tmp_path / "test_duckdb.db")
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        store.add([{"text": "First doc"}])
        results = store.query("First doc")
        assert len(results) == 1
        assert results[0]["text"] == "First doc"
        store.close()

        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        results = store.query("First doc")
        assert len(results) == 1
        assert results[0]["text"] == "First doc"
        store.close()

    def test_not_initalized(self, tmp_path):
        db_path = str(tmp_path / "test_duckdb.db")
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        store.close()

        # file exists, but we haven't added anything
        # so the indices haven't been created
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        results = store.query("First doc")
        assert len(results) == 0
        store.close()
