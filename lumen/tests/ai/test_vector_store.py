import pytest

from lumen.ai.embeddings import NumpyEmbeddings
from lumen.ai.vector_store import DuckDBVectorStore, NumpyVectorStore


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
    def store_with_three_docs(self, store):
        """
        Returns a store preloaded with three (or more) documents for convenience.
        """
        items = [
            {
                "text": "Food: $10, Drinks: $5, Total: $15. Find important info about a.dat here.",
                "metadata": {
                    "title": "receipt",
                    "department": "accounting",
                    "filename": "a.dat",
                },
            },
            {
                "text": "In the org chart, the CEO reports to the board. Another doc referencing b.csv.",
                "metadata": {
                    "title": "org_chart",
                    "department": "management",
                    "filename": "b.csv",
                },
            },
            {
                "text": "A second receipt with different details",
                "metadata": {
                    "title": "receipt",
                    "department": "accounting",
                    # no filename here, intentionally
                },
            },
        ]
        store.add(items)
        return store

    def test_filter_by_title_receipt(self, store_with_three_docs):
        filtered = store_with_three_docs.filter_by({"title": "receipt"})
        assert len(filtered) == 2

    def test_filter_by_department_management(self, store_with_three_docs):
        filtered_mgmt = store_with_three_docs.filter_by({"department": "management"})
        assert len(filtered_mgmt) == 1
        assert filtered_mgmt[0]["metadata"]["title"] == "org_chart"

    def test_filter_by_nonexistent_filter(self, store_with_three_docs):
        filtered_none = store_with_three_docs.filter_by({"title": "does_not_exist"})
        assert len(filtered_none) == 0

    def test_filter_by_limit_and_offset(self, store_with_three_docs):
        filtered_limited = store_with_three_docs.filter_by({"title": "receipt"}, limit=1)
        assert len(filtered_limited) == 1

        filtered_offset = store_with_three_docs.filter_by({"title": "receipt"}, limit=1, offset=1)
        assert len(filtered_offset) == 1
        assert filtered_offset[0]["text"] == "A second receipt with different details"

    def test_query_with_filter_title_org_chart(self, store_with_three_docs):
        query_text = "CEO reports to?"
        results = store_with_three_docs.query(query_text)
        assert len(results) >= 1
        assert results[0]["metadata"]["title"] == "org_chart"
        assert "CEO" in results[0]["text"]

        results = store_with_three_docs.query(query_text, filters={"title": "org_chart"})
        assert len(results) == 1
        assert results[0]["metadata"]["title"] == "org_chart"
        assert "CEO" in results[0]["text"]

    def test_query_with_filter_filename_single_and_multi(self, store_with_three_docs):
        # Query text that should match documents containing certain keywords
        query_text = "Find important info"

        # 1. Query with no filters
        results = store_with_three_docs.query(query_text)
        assert len(results) >= 1, "Should have at least one result without filtering."
        # Optional: if you know the first doc is 'a.dat'
        # assert results[0]["metadata"]["filename"] == "a.dat"
        # assert "important info" in results[0]["text"].lower()

        # 2. Query with a single-value filter
        #    Testing that we only retrieve documents where 'filename' == 'a.dat'
        single_filter_results = store_with_three_docs.query(
            query_text, filters={"filename": "a.dat"}
        )
        assert len(single_filter_results) >= 1, "Should find doc(s) matching filename=a.dat."
        for res in single_filter_results:
            assert res["metadata"]["filename"] == "a.dat"
            # assert "important info" in res["text"].lower()

        # 3. Query with a multi-value filter
        #    Testing that we retrieve documents where 'filename' is either 'a.dat' or 'b.csv'
        multi_filter_results = store_with_three_docs.query(
            query_text, filters={"filename": ["a.dat", "b.csv"]}, threshold=-1
        )
        assert len(multi_filter_results) >= 2, "Should find docs matching a.dat or b.csv."

    def test_query_empty_store(self, empty_store):
        results = empty_store.query("some query")
        assert results == []

    def test_filter_empty_store(self, empty_store):
        filtered = empty_store.filter_by({"key": "value"})
        assert filtered == []

    def test_delete_empty_store(self, empty_store):
        empty_store.delete([1, 2, 3])
        results = empty_store.query("some query")
        assert results == []

    def test_delete_empty_list(self, store_with_three_docs):
        original_count = len(store_with_three_docs.filter_by({}))
        store_with_three_docs.delete([])
        after_count = len(store_with_three_docs.filter_by({}))
        assert original_count == after_count

    def test_delete_specific_ids(self, store_with_three_docs):
        all_docs = store_with_three_docs.filter_by({})
        target_id = all_docs[0]["id"]
        store_with_three_docs.delete([target_id])

        remaining_docs = store_with_three_docs.filter_by({})
        remaining_ids = [doc["id"] for doc in remaining_docs]
        assert target_id not in remaining_ids

    def test_clear_store(self, store_with_three_docs):
        store_with_three_docs.clear()
        results = store_with_three_docs.filter_by({})
        assert len(results) == 0

    def test_add_docs_without_metadata(self, empty_store):
        items = [
            {"text": "Document one with no metadata"},
            {"text": "Document two with no metadata"},
        ]
        ids = empty_store.add(items)
        assert len(ids) == 2

        results = empty_store.query("Document one")
        assert len(results) > 0

    def test_add_docs_with_nonstring_metadata(self, empty_store):
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
        empty_store.add(items)

        results = empty_store.filter_by({})
        assert len(results) == 2

    def test_add_long_text_chunking(self, empty_store):
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
        ids = empty_store.add(items)

        # Should have multiple chunks, hence multiple IDs
        # (The exact number depends on chunk_size & text length.)
        assert len(ids) > 1, "Expected more than one chunk/ID due to forced chunking"

    def test_query_long_text_chunking(self, empty_store):
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
        empty_store.add(items)

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

    def test_add_multiple_large_documents(self, empty_store):
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

        ids = empty_store.add(items)
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

    def test_unique_metadata(self, store_with_three_docs):
        """
        Test the unique_metadata() method, both with and without the optional `key`.
        """
        distinct_metadata = store_with_three_docs.unique_metadata()
        assert len(distinct_metadata) == 3, f"Expected 3 distinct metadata dicts, got {len(distinct_metadata)}"

        # 2) Test retrieving distinct values for a specific key, e.g. 'title'
        titles = store_with_three_docs.unique_metadata(key="title")
        # We expect {'receipt', 'org_chart'}
        assert set(titles) == {"receipt", "org_chart"}, f"Unexpected titles: {titles}"

        # 3) Test retrieving distinct values for a key that doesn't exist
        non_existent = store_with_three_docs.unique_metadata(key="non_existent_key")
        assert non_existent == [], f"Expected empty list for non-existent key, got {non_existent}"


class TestNumpyVectorStore(VectorStoreTestKit):

    @pytest.fixture
    def store(self):
        store = NumpyVectorStore(embeddings=NumpyEmbeddings())
        store.clear()
        return store


class TestDuckDBVectorStore(VectorStoreTestKit):

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test_duckdb.db")
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        store.clear()
        return store
