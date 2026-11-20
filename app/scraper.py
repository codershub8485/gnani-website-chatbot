"""Crawl www.gnani.ai and save the cleaned text of every English page to data/pages.jsonl.

Usage:
    python -m app.scraper                 # full crawl (~1,300 pages, a few minutes)
    python -m app.scraper --limit 50      # quick partial crawl for testing

How it works:
  1. Read the URL list from the site's sitemap.xml (the authoritative list of public pages).
  2. Keep English pages only (the /ja-jp/ mirror is a translation of the same content)
     and skip anything robots.txt disallows for our user agent.
  3. Download each page politely (few workers, small delay, retries).
  4. Strip the site-wide navigation, footer and announcement banner (they repeat on every
     page and would pollute retrieval), keep headings as "## ..." markers, and save
     {url, title, description, section, text}. The menu and footer are stored once each,
     from the home page, as their own documents: the menu is the site's list of products,
     models, industries and use cases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app import config

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
# Webflow class prefixes for site-wide chrome that repeats on every page.
CHROME_CLASS_RE = re.compile(r"^(navbar|nav3_|footer|banner_|banner-)", re.I)
HIDDEN_CLASS = "w-condition-invisible"  # Webflow's "hidden by CMS condition" marker
DROP_TAGS = ["script", "style", "noscript", "svg", "iframe", "template", "form", "button", "select"]


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = config.CRAWLER_USER_AGENT
    return s


def fetch(session: requests.Session, url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def sitemap_urls(session: requests.Session) -> list[str]:
    xml = fetch(session, config.SITEMAP_URL)
    if not xml:
        sys.exit(f"Could not download {config.SITEMAP_URL}")
    root = ET.fromstring(xml.encode("utf-8"))
    urls = [loc.text.strip() for loc in root.findall("sm:url/sm:loc", SITEMAP_NS) if loc.text]
    seen, out = set(), []
    for u in urls:
        u = u.rstrip("/") or u
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def is_english(url: str) -> bool:
    path = urlparse(url).path
    return not re.match(r"^/[a-z]{2}-[a-z]{2}(/|$)", path)  # e.g. /ja-jp/...


def section_of(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.split("/")[0] if path else "home"


def _class_tokens(tag) -> list[str]:
    cls = tag.get("class") or []
    return cls if isinstance(cls, list) else str(cls).split()


def _clean_lines(text: str) -> str:
    out, seen = [], set()
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line in {"→", "->", "‍"}:
            continue
        # Drop exact repeats inside one page (logo marquees, duplicated tab labels, ...),
        # but always keep headings so the document structure survives.
        if line in seen and not line.startswith("#"):
            continue
        seen.add(line)
        out.append(line)
    return "\n".join(out)


def _text_with_headings(root) -> str:
    for level in range(1, 5):
        for h in root.find_all(f"h{level}"):
            heading = h.get_text(" ", strip=True)
            h.replace_with(f"\n{'#' * min(level + 1, 4)} {heading}\n" if heading else "")
    return _clean_lines(root.get_text("\n"))


def parse_page(html: str, url: str) -> tuple[dict, str]:
    """Return (page_record, footer_text)."""
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(strip=True) if soup.title else url
    desc_tag = soup.find("meta", attrs={"name": "description"})
    description = desc_tag.get("content", "").strip() if desc_tag else ""

    for t in soup(DROP_TAGS):
        t.decompose()
    for t in soup.find_all(class_=HIDDEN_CLASS):
        if not t.decomposed:
            t.decompose()

    # Grab the menu and footer text before removing them. They are kept only once, from
    # the home page: the menu lists every product, model, industry and use case, and the
    # footer has company links.
    site_chrome = {}
    for key, cls in (("menu", "navbar_navigation"), ("footer", "footer-v2_component")):
        el = soup.find(class_=lambda c, cls=cls: c and cls in c)
        if el:
            site_chrome[key] = _clean_lines(el.get_text("\n"))

    chrome = soup.find_all("nav") + soup.find_all(
        lambda t: any(CHROME_CLASS_RE.match(c) for c in _class_tokens(t))
    )
    for t in chrome:
        if not t.decomposed:  # may already be gone as a child of an earlier match
            t.decompose()

    body = soup.body or soup
    text = _text_with_headings(body)
    record = {
        "url": url,
        "title": title,
        "description": description,
        "section": section_of(url),
        "text": text,
    }
    return record, site_chrome


SITE_CHROME_DOCS = {
    "menu": "Gnani.ai site menu: products, models, industries, use cases, developer resources",
    "footer": "Gnani.ai site footer: products, models, resources and company links",
}


def crawl(limit: int | None = None) -> None:
    session = make_session()

    robots = urllib.robotparser.RobotFileParser(f"{config.SITE_ROOT}/robots.txt")
    robots.read()

    urls = [u for u in sitemap_urls(session) if is_english(u)]
    urls = [u for u in urls if robots.can_fetch(config.CRAWLER_USER_AGENT, u)]
    # Crawl the core product/company pages first so a --limit run is still useful.
    urls.sort(key=lambda u: (section_of(u) in {"resources", "glossary"}, u))
    if limit:
        urls = urls[:limit]
    print(f"Crawling {len(urls)} English pages from {config.SITEMAP_URL}")

    def work(u: str):
        html = fetch(session, u)
        time.sleep(config.CRAWLER_DELAY_S)
        return u, html

    records: dict[str, dict] = {}
    home_chrome: dict[str, str] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=config.CRAWLER_WORKERS) as pool:
        futures = [pool.submit(work, u) for u in urls]
        for i, fut in enumerate(as_completed(futures), 1):
            u, html = fut.result()
            if not html:
                failed.append(u)
                continue
            rec, chrome = parse_page(html, u)
            if u.rstrip("/") == config.SITE_ROOT:
                home_chrome = chrome
            if len(rec["text"].split()) >= 20:
                records[u] = rec
            if i % 100 == 0:
                print(f"  {i}/{len(urls)} fetched")

    for key, title in SITE_CHROME_DOCS.items():
        if home_chrome.get(key):
            records[key] = {
                "url": f"{config.SITE_ROOT}/",
                "title": title,
                "description": f"Site-wide {key} shown on every page of gnani.ai",
                "section": key,
                "text": home_chrome[key],
            }

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda r: r["url"])
    with open(config.PAGES_FILE, "w", encoding="utf-8") as f:
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    words = sum(len(r["text"].split()) for r in ordered)
    print(f"Saved {len(ordered)} documents ({words:,} words) to {config.PAGES_FILE}")
    if failed:
        print(f"{len(failed)} pages failed to download, e.g. {failed[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="crawl only the first N pages")
    args = ap.parse_args()
    crawl(args.limit)


if __name__ == "__main__":
    main()
