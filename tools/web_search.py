"""Live web search and page extraction."""
from __future__ import annotations
from ddgs import DDGS
from .registry import tool

@tool(description="Search the live internet for current or changing information.")
def web_search(query: str) -> str:
    """Search the live web.

    Args:
        query: Search query.
    """
    try:
        results = DDGS().text(query=query, region="in-en", safesearch="moderate", max_results=6)
    except Exception as exc:
        return f"Web search failed: {exc}"
    if not results: return f"No results found for: {query}"
    out=[]
    for i,r in enumerate(results,1):
        out.append(f"{i}. {r.get('title','Untitled')}\nURL: {r.get('href','')}\nSnippet: {r.get('body','')}")
    return "\n\n".join(out)

@tool(description="Fetch a webpage and extract readable text for summarization.")
def read_webpage(url: str) -> str:
    """Extract readable webpage text.

    Args:
        url: Full webpage URL.
    """
    try: data = DDGS().extract(url, fmt="text_plain")
    except Exception as exc: return f"Webpage extraction failed: {exc}"
    content = str(data.get("content","")).strip()
    return content[:12000] if content else f"No readable content was extracted from {url}."
