"""Web page scraping tool for the Template MCP Server.

Fetches a URL and extracts readable text content for deep research workflows.
Uses httpx for async HTTP and BeautifulSoup for HTML parsing.
"""

import re
from typing import Any, Dict

import httpx

from template_mcp_server.utils.pylogger import get_python_logger

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]

logger = get_python_logger()

_SCRAPE_TIMEOUT = 15
_MAX_CONTENT_LENGTH = 16000
_USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://redhat.com)"
)
_NOISE_TAGS = {
    "script", "style", "nav", "footer", "header", "aside",
    "form", "noscript", "iframe", "svg",
}


def _extract_text(html: str, max_length: int) -> tuple[str, str]:
    """Extract readable text and title from raw HTML."""
    if BeautifulSoup is None:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_length], ""

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    body = soup.find("body") or soup
    raw_text = body.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if len(text) > max_length:
        text = text[:max_length].rsplit("\n", 1)[0] + "\n\n[Content truncated]"

    return text, title


async def scrape_webpage(
    url: str,
    max_length: int = 8000,
) -> Dict[str, Any]:
    """Fetch a webpage and extract its readable text content.

    Use this tool to read the full content of a specific web page when
    search snippets are insufficient. Useful for deep research where
    you need detailed information from a source.

    Args:
        url: The URL to fetch and extract text from.
        max_length: Maximum characters of text content to return (100-16000).

    Returns:
        Dictionary with title, content, url, word_count, and status.
    """
    logger.info("scrape_webpage invoked", extra={"input": {"url": url}})

    try:
        if not url or not url.startswith(("http://", "https://")):
            return {
                "status": "error",
                "error": "Invalid URL — must start with http:// or https://",
                "url": url,
            }

        max_length = max(100, min(max_length, _MAX_CONTENT_LENGTH))

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(_SCRAPE_TIMEOUT),
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return {
                "status": "error",
                "error": f"Unsupported content type: {content_type}",
                "url": url,
            }

        text, title = _extract_text(response.text, max_length)
        word_count = len(text.split())

        logger.info(
            "scrape_webpage completed",
            extra={"url": url, "word_count": word_count, "title": title},
        )

        return {
            "status": "success",
            "url": url,
            "title": title,
            "content": text,
            "word_count": word_count,
        }

    except httpx.TimeoutException:
        logger.warning("scrape_webpage timeout", extra={"url": url})
        return {
            "status": "error",
            "error": f"Request timed out after {_SCRAPE_TIMEOUT}s",
            "url": url,
        }
    except httpx.HTTPStatusError as e:
        logger.warning(
            "scrape_webpage HTTP error",
            extra={"url": url, "status_code": e.response.status_code},
        )
        return {
            "status": "error",
            "error": f"HTTP {e.response.status_code}",
            "url": url,
        }
    except Exception as e:
        logger.exception("scrape_webpage failed", extra={"url": url})
        return {
            "status": "error",
            "error": str(e),
            "url": url,
        }
