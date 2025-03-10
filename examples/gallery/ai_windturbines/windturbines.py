import lumen.ai as lmai

from lumen.ai.analysis import Analysis
from lumen.sources.duckdb import DuckDBSource
from lumen.views import YdataProfilingView


class Profiling(Analysis):
    """
    Generates a profiling report for the given dataset.
    """

    def __call__(self, pipeline):
        return YdataProfilingView(pipeline=pipeline)


def duckduckgo_search(queries: list[str]) -> dict:
    """
    Perform a DuckDuckGo search for the provided queries.

    Parameters
    ----------
    queries : list[str]
        Search queries.

    Returns
    -------
    dict
        A dictionary mapping each query to a list of search results.
        Each search result is a dict containing 'title' and 'url'.
    """
    import requests

    from bs4 import BeautifulSoup

    results = {}
    for query in queries:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", {"class": "result__a"}, href=True)

        results[query] = [
            {"title": link.get_text(strip=True), "url": link["href"]} for link in links
        ]
    return results


data = [
    "https://datasets.holoviz.org/windturbines/v1/windturbines.parq",
    "https://github.com/jakevdp/data-USstates/raw/refs/heads/master/state-areas.csv",
]
analyst_agent = lmai.agents.AnalystAgent(
    template_overrides={
        "main": {"instructions": "Please focus on the outliers of the data."}
    }
)
assistant = lmai.ExplorerUI(
    data=data,
    llm=lmai.llm.OpenAI(),
    agents=[analyst_agent],
    analyses=[Profiling],
    tools=[duckduckgo_search],
)
assistant.servable()
