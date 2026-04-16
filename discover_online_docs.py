import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Discover temporary traffic management PDFs from trusted web sources,
# then optionally download them into drive_docs/_online_discovered so
# they appear in the app's document index.

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

TRUSTED_DOMAINS = [
    "austroads.com.au",
    "tmr.qld.gov.au",
    "publications.qld.gov.au",
    "legislation.qld.gov.au",
    "infrastructure.gov.au",
    "ntc.gov.au",
    "transport.nsw.gov.au",
    "roads-waterways.transport.nsw.gov.au",
    "vic.gov.au",
    "transport.vic.gov.au",
    "vicroads.vic.gov.au",
    "wa.gov.au",
    "mainroads.wa.gov.au",
    "sa.gov.au",
    "dit.sa.gov.au",
    "tas.gov.au",
    "transport.tas.gov.au",
    "act.gov.au",
    "cityservices.act.gov.au",
    "safeworkaustralia.gov.au",
    "worksafe.qld.gov.au",
    "nt.gov.au",
]

SEED_PAGES = [
    "https://austroads.com.au/publications/temporary-traffic-management",
    "https://www.tmr.qld.gov.au/business-industry/technical-standards-publications/queensland-guide-to-temporary-traffic-management",
    "https://www.publications.qld.gov.au/dataset/queensland-guide-to-temporary-traffic-management",
    "https://www.transport.nsw.gov.au/operations/roads-and-waterways/business-and-industry/partners-and-suppliers/traffic-control-at-work-sites",
    "https://www.mainroads.wa.gov.au/technical-commercial/technical-library/road-traffic-engineering/traffic-management/",
    "https://www.vicroads.vic.gov.au/business-and-industry/technical-publications",
    "https://www.dit.sa.gov.au/standards/traffic-control-devices",
    "https://www.transport.tas.gov.au/roads_and_traffic_management",
    "https://www.cityservices.act.gov.au/roads-and-paths",
    "https://www.safeworkaustralia.gov.au/doc/model-code-practice-construction-work",
    "https://www.worksafe.qld.gov.au/safety-and-prevention/hazards/working-near-traffic",
]

SEARCH_QUERIES = [
    "temporary traffic management PDF Australia",
    "AGTTM PDF",
    "QGTTM PDF",
    "MUTCD Part 3 traffic control works PDF",
    "Austroads temporary traffic management PDF",
    "traffic management design guideline PDF Australia",
    "traffic management risk assessment PDF Australia",
    "construction work risk assessment traffic control PDF",
    "safe work method statement traffic management PDF",
    "roadwork risk assessment template PDF",
    "traffic control at work sites PDF site:transport.nsw.gov.au",
    "temporary traffic management guideline PDF site:mainroads.wa.gov.au",
    "traffic management code of practice PDF site:safeworkaustralia.gov.au",
    "site:transport.nsw.gov.au traffic management manual pdf",
    "site:vicroads.vic.gov.au traffic management guideline pdf",
    "site:transport.vic.gov.au temporary traffic management pdf",
    "site:dit.sa.gov.au traffic control devices pdf",
    "site:transport.tas.gov.au traffic management pdf",
    "site:cityservices.act.gov.au temporary traffic management pdf",
    "site:nt.gov.au traffic management roadworks pdf",
]

SITE_SCOPED_QUERIES = [
    "site:{domain} temporary traffic management filetype:pdf",
    "site:{domain} traffic management design filetype:pdf",
]

SITE_DISCOVERY_DOMAINS = [
    "austroads.com.au",
    "tmr.qld.gov.au",
    "publications.qld.gov.au",
    "transport.nsw.gov.au",
    "vicroads.vic.gov.au",
    "mainroads.wa.gov.au",
    "dit.sa.gov.au",
    "transport.tas.gov.au",
    "cityservices.act.gov.au",
    "nt.gov.au",
    "safeworkaustralia.gov.au",
    "worksafe.qld.gov.au",
]

KEYWORDS = [
    "temporary traffic management",
    "traffic management design",
    "traffic control",
    "agttm",
    "qgttm",
    "mutcd",
    "austroads",
    "worksite",
    "road safety barrier",
    "risk assessment",
    "risk management",
    "safe work method statement",
    "swms",
    "job safety analysis",
    "jsa",
    "work health and safety",
]

DOWNLOAD_URL_BLOCKLIST = [
    "traffic_digest",
    "annual-report",
    "fees-schedule",
    "certificate-of-completion",
    "progress-payment",
]

JURISDICTION_ORDER = [
    "Federal / National",
    "Queensland",
    "New South Wales",
    "Victoria",
    "Western Australia",
    "South Australia",
    "Tasmania",
    "Australian Capital Territory",
    "Northern Territory",
    "Other",
]


def http_get_text(url: str, timeout: int = 8) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read()
    # Best-effort decoding
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip()
    else:
        charset = "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def normalize_url(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    # Strip fragments for dedupe
    return urllib.parse.urlunsplit((p.scheme, p.netloc.lower(), p.path, p.query, ""))


def domain_allowed(url: str) -> bool:
    try:
        host = urllib.parse.urlsplit(url).netloc.lower()
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def extract_links(html: str, base_url: str) -> list[str]:
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out = []
    for href in links:
        abs_url = urllib.parse.urljoin(base_url, href)
        if abs_url.startswith("http://") or abs_url.startswith("https://"):
            out.append(normalize_url(abs_url))
    return out


def is_pdf_url(url: str) -> bool:
    u = url.lower()
    return ".pdf" in u


def score_candidate(url: str, context_text: str = "") -> int:
    blob = (url + " " + context_text).lower()
    score = 1 if ".pdf" in blob else 0
    for kw in KEYWORDS:
        if kw in blob:
            score += 2
            continue

        # Partial keyword fallback helps when URLs use short slugs.
        tokens = [t for t in re.split(r"\W+", kw) if len(t) >= 4]
        token_hits = sum(1 for t in tokens if t in blob)
        if token_hits >= 2:
            score += 1
    return score


def search_duckduckgo(query: str) -> list[str]:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded}"
    try:
        html = http_get_text(url)
    except Exception:
        return []

    # DuckDuckGo wraps result URLs inside uddg parameter.
    results = []
    wrapped = re.findall(r'href=["\']https?://duckduckgo\.com/l/\?[^"\']*uddg=([^"\'&]+)', html, flags=re.IGNORECASE)
    for encoded_target in wrapped:
        try:
            target = urllib.parse.unquote(encoded_target)
            target = normalize_url(target)
            if target.startswith("http"):
                results.append(target)
        except Exception:
            continue
    return results[:12]


def search_bing(query: str) -> list[str]:
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.bing.com/search?q={encoded}"
    try:
        html = http_get_text(url)
    except Exception:
        return []

    results = []
    links = re.findall(r'href=["\'](https?://[^"\']+)["\']', html, flags=re.IGNORECASE)
    for link in links:
        link = normalize_url(link)
        if not link.startswith("http"):
            continue
        if "bing.com" in urllib.parse.urlsplit(link).netloc.lower():
            continue
        results.append(link)
    return results[:12]


def search_web(query: str) -> list[str]:
    # Aggregate from multiple public result pages to improve resilience.
    seen = set()
    out = []
    for fn in (search_duckduckgo, search_bing):
        for link in fn(query):
            key = link.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(link)
    return out


def candidate_key(url: str) -> str:
    return normalize_url(url).lower()


def classify_candidate(url: str, context_text: str = "") -> tuple[str, str]:
    text = f"{url} {context_text}".lower()
    host = urllib.parse.urlsplit(url).netloc.lower()

    if any(token in text for token in ["austroads", "safeworkaustralia", "infrastructure.gov.au", "ntc.gov.au"]):
        return "Federal / National", host
    if any(token in text for token in ["qld", "queensland", "tmr.qld.gov.au", "publications.qld.gov.au", "worksafe.qld.gov.au", "mutcd"]):
        return "Queensland", host
    if any(token in text for token in ["nsw", "new south wales", "transport.nsw.gov.au", "roads-waterways.transport.nsw.gov.au"]):
        return "New South Wales", host
    if any(token in text for token in ["victoria", "vicroads", "transport.vic.gov.au", "vic.gov.au"]):
        return "Victoria", host
    if any(token in text for token in ["western australia", "mainroads.wa.gov.au", "wa.gov.au"]):
        return "Western Australia", host
    if any(token in text for token in ["south australia", "dit.sa.gov.au", "sa.gov.au"]):
        return "South Australia", host
    if any(token in text for token in ["tasmania", "transport.tas.gov.au"]):
        return "Tasmania", host
    if any(token in text for token in ["canberra", "cityservices.act.gov.au", "australian capital territory"]):
        return "Australian Capital Territory", host
    if any(token in text for token in ["northern territory", "nt.gov.au"]):
        return "Northern Territory", host
    return "Other", host


def add_candidate(candidates: dict, link: str, source: str, score: int, **extra) -> None:
    key = candidate_key(link)
    existing = candidates.get(key)
    if existing:
        existing["score"] = max(existing["score"], score)
        return

    jurisdiction, authority = classify_candidate(link, json.dumps(extra, ensure_ascii=False))

    item = {
        "url": link,
        "source": source,
        "score": score,
        "jurisdiction": jurisdiction,
        "authority": authority,
    }
    item.update(extra)
    candidates[key] = item


def discover_pdfs_from_page(page_url: str, source: str, min_score: int) -> list[tuple[str, int]]:
    try:
        html = http_get_text(page_url)
    except Exception:
        return []

    found = []
    for link in extract_links(html, page_url):
        if not domain_allowed(link):
            continue
        if not is_pdf_url(link):
            continue
        score = score_candidate(link, page_url)
        if score >= min_score:
            found.append((link, score))
    return found


def discover_online_candidates(min_score: int = 1, max_candidates: int = 300) -> list[dict]:
    candidates = {}
    crawl_pages = []
    crawled_pages = set()

    # 1) Crawl seed pages for direct PDF links.
    for page in SEED_PAGES:
        try:
            html = http_get_text(page)
        except Exception:
            continue

        for link in extract_links(html, page):
            if not domain_allowed(link):
                continue

            if not is_pdf_url(link):
                crawl_pages.append(link)
                continue

            score = score_candidate(link)
            if is_pdf_url(link) and score >= min_score:
                add_candidate(candidates, link, "seed_page", score, seed=page)

    # 1b) One-hop crawl of trusted pages discovered from seed pages.
    for page in crawl_pages[:80]:
        if page in crawled_pages:
            continue
        crawled_pages.add(page)
        for pdf_link, score in discover_pdfs_from_page(page, "seed_subpage", min_score):
            add_candidate(candidates, pdf_link, "seed_subpage", score, from_page=page)

    # 2) Search web and keep trusted-domain PDF links.
    all_queries = list(SEARCH_QUERIES)
    for domain in SITE_DISCOVERY_DOMAINS:
        for tmpl in SITE_SCOPED_QUERIES:
            all_queries.append(tmpl.format(domain=domain))

    # Keep query count bounded so startup discovery stays responsive.
    seen_queries = set()
    all_queries = [q for q in all_queries if not (q in seen_queries or seen_queries.add(q))][:36]

    for q in all_queries:
        for link in search_web(q):
            if not domain_allowed(link):
                continue

            if is_pdf_url(link):
                score = score_candidate(link, q)
                if score < min_score:
                    continue
                add_candidate(candidates, link, "search", score, query=q)
            else:
                if link in crawled_pages:
                    continue
                crawled_pages.add(link)
                for pdf_link, score in discover_pdfs_from_page(link, "search_subpage", min_score):
                    add_candidate(candidates, pdf_link, "search_subpage", score, query=q, from_page=link)

            if len(candidates) >= max_candidates:
                break

        if len(candidates) >= max_candidates:
            break
        time.sleep(0.2)

    # Sorted by score desc, then url
    items = sorted(candidates.values(), key=lambda x: (-x["score"], x["url"]))
    return items


def safe_filename_from_url(url: str) -> str:
    path = urllib.parse.urlsplit(url).path
    name = os.path.basename(path) or "document.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    # Keep ASCII-ish filename
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name


def jurisdiction_folder_name(jurisdiction: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", jurisdiction.strip())
    return safe or "Other"


def build_existing_name_index(drive_docs_root: Path) -> set[str]:
    names = set()
    if not drive_docs_root.exists():
        return names
    for p in drive_docs_root.rglob("*.pdf"):
        names.add(p.name.lower())
    return names


def build_existing_hash_index(drive_docs_root: Path) -> set[str]:
    hashes = set()
    if not drive_docs_root.exists():
        return hashes
    for p in drive_docs_root.rglob("*.pdf"):
        try:
            hashes.add(hashlib.sha1(p.read_bytes()).hexdigest())
        except Exception:
            continue
    return hashes


def order_candidates_balanced(candidates: list[dict]) -> list[dict]:
    grouped = {}
    for item in candidates:
        grouped.setdefault(item.get("jurisdiction", "Other"), []).append(item)

    for items in grouped.values():
        items.sort(key=lambda x: (-x.get("score", 0), x.get("url", "")))

    ordered_jurisdictions = [j for j in JURISDICTION_ORDER if j in grouped]
    ordered_jurisdictions += [j for j in grouped.keys() if j not in ordered_jurisdictions]

    merged = []
    while True:
        progressed = False
        for j in ordered_jurisdictions:
            bucket = grouped.get(j, [])
            if bucket:
                merged.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return merged


def download_candidates(candidates: list[dict], out_dir: Path, drive_docs_root: Path, max_download: int, interactive: bool, exclude_jurisdictions: set[str] | None = None, max_per_jurisdiction: int = 6) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    existing_hashes = build_existing_hash_index(drive_docs_root)

    filtered_candidates = []
    excluded = {x.lower() for x in (exclude_jurisdictions or set())}
    for item in candidates:
        jurisdiction = (item.get("jurisdiction") or "Other").lower()
        if jurisdiction in excluded:
            continue
        filtered_candidates.append(item)

    candidate_stream = order_candidates_balanced(filtered_candidates)
    jurisdiction_counts = {}

    count = 0
    for item in candidate_stream:
        if count >= max_download:
            break

        url = item["url"]
        url_lower = url.lower()
        if any(token in url_lower for token in DOWNLOAD_URL_BLOCKLIST):
            continue

        jurisdiction = item.get("jurisdiction") or "Other"
        if jurisdiction_counts.get(jurisdiction, 0) >= max_per_jurisdiction:
            continue

        if interactive:
            print(f"\nCandidate ({item['score']}): {url}")
            choice = input("Download this file? [y/N/q]: ").strip().lower()
            if choice == "q":
                break
            if choice != "y":
                continue

        fname = safe_filename_from_url(url)
        target_dir = out_dir / jurisdiction_folder_name(jurisdiction)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / fname

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                data = resp.read()
            # Keep only PDF-like responses
            if b"%PDF" not in data[:2048] and "pdf" not in content_type:
                continue
            digest = hashlib.sha1(data).hexdigest()
            if digest in existing_hashes:
                continue

            if target.exists():
                stem = target.stem
                suffix = target.suffix
                i = 1
                while target.exists():
                    target = target_dir / f"{stem}_{i}{suffix}"
                    i += 1

            target.write_bytes(data)
        except Exception:
            continue

        existing_hashes.add(digest)
        jurisdiction_counts[jurisdiction] = jurisdiction_counts.get(jurisdiction, 0) + 1
        downloaded.append({
            "url": url,
            "saved_to": str(target),
            "score": item["score"],
            "jurisdiction": jurisdiction,
        })
        count += 1
        time.sleep(0.2)

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover online temporary traffic management PDFs.")
    parser.add_argument("--min-score", type=int, default=1, help="Minimum relevance score (default: 1)")
    parser.add_argument("--download", action="store_true", help="Download discovered PDFs into drive_docs/_online_discovered")
    parser.add_argument("--interactive", action="store_true", help="Ask approval for each file before downloading")
    parser.add_argument("--max-download", type=int, default=15, help="Max number of PDFs to download when --download is used")
    parser.add_argument("--max-per-jurisdiction", type=int, default=6, help="Max files to download per jurisdiction in one run")
    parser.add_argument("--exclude-jurisdiction", action="append", default=[], help="Jurisdiction to exclude from downloads (repeatable)")
    parser.add_argument("--drive-docs", default="./drive_docs", help="Drive docs root path (default: ./drive_docs)")
    parser.add_argument("--report", default="online_discovery_report.json", help="JSON report path")
    args = parser.parse_args()

    candidates = discover_online_candidates(min_score=args.min_score)

    report = {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "grouped_candidates": {},
        "downloaded_count": 0,
        "downloaded": [],
    }

    grouped = {name: [] for name in JURISDICTION_ORDER}
    for item in candidates:
        grouped.setdefault(item.get("jurisdiction", "Other"), []).append(item)
    report["grouped_candidates"] = {key: value for key, value in grouped.items() if value}

    if args.download and candidates:
        drive_docs_root = Path(args.drive_docs)
        out_dir = drive_docs_root / "_online_discovered"
        downloaded = download_candidates(
            candidates,
            out_dir,
            drive_docs_root,
            args.max_download,
            args.interactive,
            exclude_jurisdictions=set(args.exclude_jurisdiction),
            max_per_jurisdiction=args.max_per_jurisdiction,
        )
        report["downloaded"] = downloaded
        report["downloaded_count"] = len(downloaded)

    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Candidates found: {report['candidate_count']}")
    if args.download:
        print(f"Downloaded: {report['downloaded_count']}")
        print(f"Download folder: {(Path(args.drive_docs) / '_online_discovered').resolve()}")
    print(f"Report: {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
