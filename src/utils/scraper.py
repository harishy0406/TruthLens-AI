"""
scraper.py
──────────
Fetches up to 20 recent news articles from public RSS feeds
and optional Google News API. No API key required for RSS mode.

Usage:
    from src.utils.scraper import fetch_news
    articles = fetch_news(n=20)
"""

import time
import random
import logging
from datetime import datetime
from typing import List, Dict

import requests

logger = logging.getLogger(__name__)

# ── Public RSS feeds (no API key needed) ────────────────────────────────────────
RSS_FEEDS = [
    ("Reuters",          "https://feeds.reuters.com/reuters/topNews"),
    ("BBC News",         "http://feeds.bbci.co.uk/news/rss.xml"),
    ("Al Jazeera",       "https://www.aljazeera.com/xml/rss/all.xml"),
    ("AP News",          "https://rsshub.app/apnews/topics/apf-topnews"),
    ("The Guardian",     "https://www.theguardian.com/world/rss"),
    ("NPR News",         "https://feeds.npr.org/1001/rss.xml"),
    ("CNN",              "http://rss.cnn.com/rss/edition.rss"),
    ("ABC News",         "https://feeds.abcnews.com/abcnews/topstories"),
    ("NBC News",         "https://feeds.nbcnews.com/nbcnews/public/news"),
    ("USA Today",        "http://rssfeeds.usatoday.com/usatoday-NewsTopStories"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _parse_rss(feed_url: str, source_name: str, timeout: int = 8) -> List[Dict]:
    """Parse a single RSS feed. Returns list of article dicts."""
    articles = []
    try:
        import feedparser
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:5]:
            title = entry.get("title", "").strip()
            link  = entry.get("link", "").strip()
            if not title or not link:
                continue

            # Parse date
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub:
                date_str = datetime(*pub[:6]).strftime("%Y-%m-%d %H:%M")
            else:
                date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

            # Summary / description
            summary = entry.get("summary", "") or entry.get("description", "")
            # Strip HTML tags from summary
            import re
            summary = re.sub(r"<[^>]+>", "", summary).strip()[:500]

            articles.append({
                "title":       title,
                "source":      source_name,
                "url":         link,
                "published_at": date_str,
                "summary":     summary or title,
            })
    except Exception as e:
        logger.warning(f"[scraper] Failed to parse {source_name}: {e}")
    return articles


def fetch_news(n: int = 20, shuffle: bool = True) -> List[Dict]:
    """
    Fetch up to `n` news articles from multiple RSS feeds.

    Parameters
    ----------
    n       : Maximum number of articles to return.
    shuffle : Whether to shuffle across sources for variety.

    Returns
    -------
    List of dicts: { title, source, url, published_at, summary }
    """
    all_articles: List[Dict] = []

    feeds = RSS_FEEDS.copy()
    if shuffle:
        random.shuffle(feeds)

    for source_name, feed_url in feeds:
        if len(all_articles) >= n:
            break
        batch = _parse_rss(feed_url, source_name)
        all_articles.extend(batch)
        time.sleep(0.3)  # polite crawling

    if shuffle:
        random.shuffle(all_articles)

    result = all_articles[:n]
    print(f"[scraper] Fetched {len(result)} articles from {len(feeds)} feeds.")
    return result


def fetch_article_text(url: str, timeout: int = 10) -> str:
    """Attempt to fetch the full article text from a URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        # Extract main paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text[:3000]
    except Exception as e:
        logger.warning(f"[scraper] Could not fetch article text: {e}")
        return ""


if __name__ == "__main__":
    articles = fetch_news(n=20)
    for i, a in enumerate(articles, 1):
        print(f"{i:2d}. [{a['source']}] {a['title'][:80]}")
