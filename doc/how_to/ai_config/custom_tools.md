# Custom Tools

Tools extend the capabilities of LLMs/agents by enabling them to access external data, perform specialized tasks, and interact dynamically with the environment.

Besides the built-in tools, users can create custom tools to perform specific actions tailored to their use case and/or domain.

For example, a user may want to create a `duckduckgo_search` tool to search for information on the internet.

To do so, the user simply has to:

1. Create a function with the desired functionality
2. Provide type hints for the input arguments so the LLM can provide valid inputs to the function
3. Write a docstring to describe the function's purpose so the LLM can know when to apply it
4. Add the function to the `tools` list

```python
def duckduckgo_search(queries: list[str]) -> dict:
    """
    Perform a DuckDuckGo search for the provided queries.

    Arguments
    ---------
    queries : list[str]
        Search queries.

    Returns
    -------
    dict
        Search results with the query as keys and lists of titles and links as values.
    """
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

tools = [duckduckgo_search]
ui = lmai.ExplorerUI(tools=tools)
```

By sharing the `duckduckgo_search` tool with the `Coordinator`, the subsequent `Agent`s can digest the tools' results to provide more accurate and relevant information to the user.

Another example is a `get_wiki` tool that fetches summaries of Wikipedia articles.

```python
def get_wiki(articles: list[str]) -> str:
    wiki = wikipediaapi.Wikipedia("lumen-assistant", language="en")
    out = ""
    for article in articles:
        page = wiki.page(article)
        if page.exists():
            out += f"{article}:\n{page.summary}\n\n"
        else:
            out += f"The article '{article}' does not exist.\n"
    return out

tools = [get_wiki]
agents = [lmai.agents.ChatAgent(prompts={"main": {"tools": tools}})]
```
