import asyncio

from pathlib import Path

import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.embeddings import Embeddings, NumpyEmbeddings, OpenAIEmbeddings
from lumen.ai.vector_store import DuckDBVectorStore, NumpyVectorStore


class VectorStoreTestKit:
    """
    A base class (test kit) that provides the *common* tests and fixture definitions.
    """

    @pytest.fixture
    async def store(self):
        """
        Must be overridden in the subclass to return a fresh store instance.
        """
        raise NotImplementedError("Subclasses must override `store` fixture")

    @pytest.fixture
    async def empty_store(self, store):
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
        results = await store_with_three_docs.query(query_text)
        assert len(results) >= 1
        assert results[0]["metadata"]["title"] == "org_chart"
        assert "CEO" in results[0]["text"]

        results = await store_with_three_docs.query(query_text, filters={"title": "org_chart"})
        assert len(results) == 1
        assert results[0]["metadata"]["title"] == "org_chart"
        assert "CEO" in results[0]["text"]

    @pytest.mark.asyncio
    async def test_query_1_threshold(self, store_with_three_docs):
        text = "Food: $10, Drinks: $5, Total: $15"
        metadata = {"title": "receipt", "department": "accounting"}
        stored_embedding_text = store_with_three_docs._join_text_and_metadata(text, metadata)
        results = await store_with_three_docs.query(stored_embedding_text, threshold=0.99)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_empty_store(self, empty_store):
        results = await empty_store.query("some query")
        assert results == []

    @pytest.mark.asyncio
    async def test_filter_empty_store(self, empty_store):
        filtered = empty_store.filter_by({"key": "value"})
        assert filtered == []

    @pytest.mark.asyncio
    async def test_delete_empty_store(self, empty_store):
        empty_store.delete([1, 2, 3])
        results = await empty_store.query("some query")
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

        results = await empty_store.query("Document one")
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
        Verifies that adding a document with text longer than `chunk_size` works correctly.
        """
        # Set a small chunk size
        empty_store.chunk_size = 50

        # Create a long text with clear semantic breaks to encourage chunking
        section1 = "UNIQUE_TERM_ONE content for first section. " * 10
        section2 = "UNIQUE_TERM_TWO content for second section. " * 10
        long_text = section1 + "\n\n" + section2

        items = [
            {
                "text": long_text,
                "metadata": {"title": "long_document"},
            }
        ]
        ids = await empty_store.add(items)

        # Verify documents were added successfully
        assert len(ids) >= 1, "Should have at least one chunk/ID"

        # Verify we can retrieve content from different sections using exact terms from the text
        results1 = await empty_store.query("UNIQUE_TERM_ONE")
        assert len(results1) > 0, "Should be able to retrieve content with UNIQUE_TERM_ONE"

        results2 = await empty_store.query("UNIQUE_TERM_TWO")
        assert len(results2) > 0, "Should be able to retrieve content with UNIQUE_TERM_TWO"

    @pytest.mark.asyncio
    async def test_query_long_text_chunking(self, empty_store):
        """
        Verifies querying a store containing a large text still returns sensible results.
        """
        empty_store.chunk_size = 60

        # Create long text with distinct sections that should be semantically separable
        section1 = "APPLE_XYZ_123 is a unique identifier for fruits. " * 15
        section2 = "BANANA_XYZ_456 is a unique identifier for other fruits. " * 15
        long_text = section1 + "\n\n" + section2

        items = [
            {
                "text": long_text,
                "metadata": {"title": "very_large_document"},
            }
        ]
        await empty_store.add(items)

        # Query for unique terms in each section
        results_apples = await empty_store.query("APPLE_XYZ_123")
        assert len(results_apples) > 0, "Should have at least one chunk matching APPLE_XYZ_123"

        results_bananas = await empty_store.query("BANANA_XYZ_456")
        assert len(results_bananas) > 0, "Should have at least one chunk matching BANANA_XYZ_456"

    @pytest.mark.asyncio
    async def test_add_multiple_large_documents(self, empty_store):
        """
        Verifies behavior when multiple large documents are added.
        """
        # Set a reasonable chunk size
        empty_store.chunk_size = 100

        # Create documents with distinct content and unique identifiers
        doc1 = "PYTHON_CODE_789 is a unique identifier for programming content. " * 10
        doc2 = "SQL_DATABASE_ABC is a unique identifier for database content. " * 10

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
        # Verify documents were added
        assert len(ids) >= 2, "Expected at least one chunk per document"

        # Query for content from doc2 using the unique identifier
        results = await empty_store.query("SQL_DATABASE_ABC")
        assert len(results) > 0
        # Expect at least 1 result from doc2
        found_doc2 = any("large_document_2" in r["metadata"].get("title", "") for r in results)
        assert found_doc2, "Expected to find at least one chunk belonging to doc2"

        # Query for content from doc1 using the unique identifier
        results = await empty_store.query("PYTHON_CODE_789")
        assert len(results) > 0
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

        results = await empty_store.query("A new document")
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
        results1 = await empty_store.query("Hello Python world!")
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

        # Verify metadata was updated (key3 was removed)
        results = empty_store.filter_by({"key1": "value1"})
        assert len(results) == 1, "Should still have only one item"
        assert len(results[0]["metadata"]) == 2, "Should now have only two metadata keys"
        assert "key3" not in results[0]["metadata"], "key3 should be removed"

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self, empty_store):
        """
        Verifies that upsert handles empty input gracefully.
        """
        ids = await empty_store.upsert([])
        assert ids == [], "Should return empty list of IDs"

        # Store should remain empty
        all_docs = empty_store.filter_by({})
        assert len(all_docs) == 0, "Store should still be empty"

    @pytest.mark.asyncio
    async def test_upsert_long_content_no_duplication(self, empty_store):
        """
        Verifies that upsert doesn't create duplicates when adding the same long text
        that gets chunked.
        """
        # Set a chunk size
        empty_store.chunk_size = 100

        # Create a long text with meaningful semantic sections and unique identifiers
        paragraph1 = "MACHINE_LEARNING_XYZ is a unique identifier in this paragraph. " * 5
        paragraph2 = "NLP_TRANSFORMER_123 is a unique identifier in this paragraph. " * 5
        long_text = paragraph1 + "\n\n" + paragraph2

        metadata = {"source": "test", "type": "long_document"}

        # First upsert
        item = {"text": long_text, "metadata": metadata}
        ids1 = await empty_store.upsert([item])

        # Verify document was added
        assert len(ids1) >= 1, "Text should be added with at least one chunk"

        # Verify we can query for unique terms
        results1 = await empty_store.query("MACHINE_LEARNING_XYZ")
        assert len(results1) > 0, "Should be able to retrieve content with unique identifier"

        # Record count after first upsert
        count_after_first = len(empty_store)

        # Perform second upsert with the same content
        await empty_store.upsert([item])
        count_after_second = len(empty_store)

        # Verify no new entries were created
        assert count_after_first == count_after_second, "No new items should be added on second upsert"

    async def test_add_directory(self, empty_store):
        """
        Verifies that adding a directory of files works correctly.
        """
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg")
        dir_path = Path(__file__).parent / "test_dir"

        # Add the directory to the store
        ids = await empty_store.add_directory(dir_path, metadata={"version": 1}, upsert=True)
        await asyncio.sleep(0.1)
        assert len(ids) > 0, "Should add at least one document"

        # Query for a specific term in the added documents
        assert len(await empty_store.query("Sed elementum")) > 0, "Should find the term in the added documents"

        # Try upserting again
        same_ids = await empty_store.add_directory(dir_path, metadata={"version": 1}, upsert=True)
        assert set(ids) == set(same_ids), "Should return the same IDs when upserting identical content"

        # Increment version
        new_ids = await empty_store.add_directory(dir_path, metadata={"version": 2}, upsert=True)
        assert len(new_ids) > 0, "Should add at least one document"
        assert len(set(new_ids) & set(ids)) == 0, "Should return different IDs when version changes"


class TestNumpyVectorStore(VectorStoreTestKit):

    @pytest.fixture
    async def store(self):
        store = NumpyVectorStore(embeddings=NumpyEmbeddings())
        store.clear()
        return store


@pytest.mark.xdist_group("vss")
class TestDuckDBVectorStore(VectorStoreTestKit):

    @pytest.fixture
    async def store(self, tmp_path) -> DuckDBVectorStore:
        db_path = str(tmp_path / "test_duckdb.db")
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        store.clear()
        return store

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        db_path = str(tmp_path / "test_duckdb.db")
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        await store.add([{"text": "First doc"}])
        results = await store.query("First doc")
        assert len(results) == 1
        assert results[0]["text"] == "First doc"
        store.close()

        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        results = await store.query("First doc")
        assert len(results) == 1
        assert results[0]["text"] == "First doc"
        store.close()

    @pytest.mark.asyncio
    async def test_not_initalized(self, tmp_path):
        db_path = str(tmp_path / "test_duckdb.db")
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        store.close()

        # file exists, but we haven't added anything
        # so the indices haven't been created
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        results = await store.query("First doc")
        assert len(results) == 0
        store.close()

    async def test_check_embeddings_consistency(self, tmp_path):
        db_path = str(tmp_path / "test_duckdb.db")
        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        await store.add([{"text": "First doc"}])
        store.close()

        store = DuckDBVectorStore(uri=db_path, embeddings=NumpyEmbeddings())
        assert len(await store.query("First doc")) == 1
        store.close()

        with pytest.raises(ValueError, match="Provided embeddings class"):
            DuckDBVectorStore(uri=db_path, embeddings=Embeddings())

    @pytest.mark.asyncio
    async def test_api_key_not_stored_in_metadata(self, tmp_path):
        """Verifies that api_key parameter is not included in stored embeddings metadata."""
        import json
        
        db_path = str(tmp_path / "test_duckdb.db")
        
        embeddings = OpenAIEmbeddings(api_key="sk-test-secret-key-12345")
        store = DuckDBVectorStore(uri=db_path, embeddings=embeddings)
        store._setup_database(1)

        metadata_result = store.connection.execute(
            "SELECT value FROM vector_store_metadata WHERE key = 'embeddings';"
        ).fetchone()
        assert metadata_result is not None, "Embeddings metadata should be stored"

        metadata = json.loads(metadata_result[0])
        params = metadata.get("params", {})
        assert "api_key" not in params, "api_key should not be stored in metadata"
        store.close()


# Sample readme content for testing (mimics real-world metadata files)
SAMPLE_README = """# Population - Data package

This data package contains the data that powers the chart ["Population"](https://ourworldindata.org/grapher/population-with-un-projections?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website.

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For normal countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The remaining columns are the data columns, each of which is a time series. If the CSV data is downloaded using the "full data" option, then each column corresponds to one time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data columns are transformed depending on the chart type and thus the association with the time series might not be as straightforward.

## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

### How we process data at Our World In Data
All data and visualizations on Our World in Data rely on data sourced from one or several original data providers. Preparing this original data involves several processing steps. Depending on the data, this can include standardizing country names and world region definitions, converting units, calculating derived indicators such as per capita measures, as well as adding or adapting metadata such as the name or the description given to an indicator.
[Read about our data pipeline](https://docs.owid.io/projects/etl/)

## Detailed information about each time series


## Population, total - UN WPP
De facto total population in a country, area or region as of 1 July of the year indicated.
Last updated: July 12, 2024
Date range: 1950-2023
Unit: people


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:
UN, World Population Prospects (2024) - processed by Our World in Data

#### Full citation
UN, World Population Prospects (2024) - processed by Our World in Data. "Population, total - UN WPP" [dataset]. United Nations, "World Population Prospects" [original data].
Source: UN, World Population Prospects (2024) - processed by Our World In Data

### Source

#### United Nations - World Population Prospects
Retrieved on: 2024-07-11
Retrieved from: https://population.un.org/wpp/downloads/


## Population, medium projection - UN WPP
De facto total population in a country, area or region as of 1 July of the year indicated.  Projections from 2024 onwards are based on the UN's medium scenario.
Last updated: July 12, 2024
Date range: 2024-2100
Unit: people


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:
UN, World Population Prospects (2024) - processed by Our World in Data

#### Full citation
UN, World Population Prospects (2024) - processed by Our World in Data. "Population, medium projection - UN WPP" [dataset]. United Nations, "World Population Prospects" [original data].
Source: UN, World Population Prospects (2024) - processed by Our World In Data

### Source

#### United Nations - World Population Prospects
Retrieved on: 2024-07-11
Retrieved from: https://population.un.org/wpp/downloads/
"""


class TestDocumentUploadFlow:
    """
    Tests that mimic the document upload flow from controls.py to verify
    that documents are properly chunked and queryable.
    """

    @pytest.fixture
    async def store(self):
        # Use smaller chunk size to ensure documents get chunked
        store = NumpyVectorStore(embeddings=NumpyEmbeddings(), chunk_size=512)
        store.clear()
        return store

    @pytest.mark.asyncio
    async def test_readme_upload_creates_chunks(self, store):
        """
        Verifies that uploading a readme file creates multiple chunks,
        not a single large entry.
        """
        # Mimic what controls.py does: upsert with filename and type metadata
        doc_entry = {
            "text": SAMPLE_README,
            "metadata": {
                "filename": "readme.md",
                "type": "document",
            },
        }
        
        ids = await store.upsert([doc_entry])
        
        # Should have created multiple chunks (readme is ~4000 chars, chunk_size=512 tokens)
        assert len(ids) > 1, f"Expected multiple chunks, got {len(ids)} chunk(s). Document length: {len(SAMPLE_README)} chars"
        
        # All chunks should have the same metadata
        all_docs = store.filter_by({"filename": "readme.md"})
        assert len(all_docs) == len(ids), "All chunks should have filename metadata"
        
        for doc in all_docs:
            assert doc["metadata"]["type"] == "document"
            assert doc["metadata"]["filename"] == "readme.md"

    @pytest.mark.asyncio
    async def test_query_returns_relevant_chunks_not_whole_doc(self, store):
        """
        Verifies that querying returns relevant chunks, not the entire document.
        """
        doc_entry = {
            "text": SAMPLE_README,
            "metadata": {
                "filename": "readme.md",
                "type": "document",
            },
        }
        
        await store.upsert([doc_entry])
        
        # Query for specific content
        results = await store.query(
            "How to cite this data",
            top_k=3,
            filters={"type": "document", "filename": "readme.md"}
        )
        
        assert len(results) > 0, "Should find at least one result"
        
        # Each result should be a chunk, not the whole document
        for result in results:
            chunk_text = result["text"]
            assert len(chunk_text) < len(SAMPLE_README), (
                f"Chunk length ({len(chunk_text)}) should be less than full doc ({len(SAMPLE_README)})"
            )

    @pytest.mark.asyncio
    async def test_query_different_sections_returns_different_chunks(self, store):
        """
        Verifies that querying different topics returns different relevant chunks.
        """
        doc_entry = {
            "text": SAMPLE_README,
            "metadata": {
                "filename": "readme.md",
                "type": "document",
            },
        }
        
        await store.upsert([doc_entry])
        
        # Query for CSV structure
        csv_results = await store.query(
            "CSV structure Entity Code columns",
            top_k=1,
            filters={"type": "document"}
        )
        
        # Query for citation
        citation_results = await store.query(
            "How to cite UN World Population Prospects",
            top_k=1,
            filters={"type": "document"}
        )
        
        assert len(csv_results) > 0, "Should find CSV structure chunk"
        assert len(citation_results) > 0, "Should find citation chunk"
        
        # The chunks should be different (different parts of the document)
        csv_chunk = csv_results[0]["text"]
        citation_chunk = citation_results[0]["text"]
        
        # They might overlap somewhat, but shouldn't be identical
        assert csv_chunk != citation_chunk, "Different queries should return different chunks"

    @pytest.mark.asyncio
    async def test_chunk_size_affects_number_of_chunks(self):
        """
        Verifies that smaller chunk_size creates more chunks.
        """
        doc_entry = {
            "text": SAMPLE_README,
            "metadata": {
                "filename": "readme.md",
                "type": "document",
            },
        }
        
        # Create store with small chunk size
        small_store = NumpyVectorStore(embeddings=NumpyEmbeddings(), chunk_size=256)
        ids_small = await small_store.upsert([doc_entry])
        small_chunk_count = len(ids_small)
        
        # Create store with larger chunk size
        large_store = NumpyVectorStore(embeddings=NumpyEmbeddings(), chunk_size=2048)
        ids_large = await large_store.upsert([doc_entry])
        large_chunk_count = len(ids_large)
        
        assert small_chunk_count > large_chunk_count, (
            f"Smaller chunk_size ({small_chunk_count} chunks) should create more chunks "
            f"than larger chunk_size ({large_chunk_count} chunks)"
        )

    @pytest.mark.asyncio
    async def test_upsert_same_document_twice_no_duplicates(self, store):
        """
        Verifies that upserting the same document twice doesn't create duplicates.
        """
        doc_entry = {
            "text": SAMPLE_README,
            "metadata": {
                "filename": "readme.md",
                "type": "document",
            },
        }
        
        # First upsert
        ids1 = await store.upsert([doc_entry])
        count_after_first = len(store)
        
        # Second upsert (same content)
        ids2 = await store.upsert([doc_entry])
        count_after_second = len(store)
        
        assert count_after_first == count_after_second, (
            f"Second upsert should not create duplicates. "
            f"First: {count_after_first}, Second: {count_after_second}"
        )
        assert set(ids1) == set(ids2), "Should return same IDs on duplicate upsert"
