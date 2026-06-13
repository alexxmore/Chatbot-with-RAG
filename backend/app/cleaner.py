"""HTML/ASPX cleaning pipeline: trafilatura + BeautifulSoup.

Two SharePoint page formats are supported:

1. ASPX (raw source) — content inside <!--[if gte mso 9]--> conditional comment
   as mso:CanvasContent1 HTML-entity-encoded HTML. Extracted via regex → BeautifulSoup.

2. HTML (browser-saved, JS-rendered) — content embedded in a 900KB+ inline script
   as JSON-encoded strings. Extracted by finding all Ukrainian text segments > 20 chars
   from the large script, deduplicating, and joining in order.
"""
import html
import re

import trafilatura
from bs4 import BeautifulSoup

# Matches "5.Що робити..." – a numbered FAQ question in a <p><strong> tag
_NUMBERED_Q_RE = re.compile(r"^\d+\.")

# Matches the full mso:CanvasContent1 attribute value (HTML-entity encoded HTML)
_CANVAS_RE = re.compile(
    r"<mso:CanvasContent1[^>]*>(.*?)</mso:CanvasContent1>",
    re.DOTALL | re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

# Ukrainian text segments: start with Cyrillic, ≥ 20 mixed chars
# Used to extract content from JS-rendered SharePoint HTML pages
_UK_SEGMENT_RE = re.compile(
    r"[А-ЯҐЄІЇа-яґєії][А-ЯҐЄІЇа-яґєіїa-zA-Z\s\-,.():;!?«»'\"—–/\d]{20,}"
)
# Minimum script size to be considered a JS-rendered SharePoint page (bytes)
_JS_PAGE_SCRIPT_MIN = 300_000

# A Cyrillic-led segment can greedily swallow the adjacent SharePoint nav/auth JSON
# blob (…aspx","layoutsUrl":"…","userPuid":"…"). Cut the segment at the first
# JSON key/value boundary so author names, userPuid and other metadata never index.
_JSON_TAIL_RE = re.compile(r'"\s*,\s*"\w+"\s*:')


def extract_text(file_content: str, filename: str = "") -> tuple[str, str]:
    """Return (title, clean_text) from HTML or SharePoint ASPX content."""
    # Extract <title> via regex (works even inside conditional comments)
    title = ""
    m = _TITLE_RE.search(file_content)
    if m:
        title = html.unescape(m.group(1)).strip()

    # Path 1: SharePoint ASPX — CanvasContent1 in IE conditional comment
    cm = _CANVAS_RE.search(file_content)
    if cm:
        raw_entities = cm.group(1)
        decoded = html.unescape(raw_entities)
        text = _extract_from_sharepoint_html(decoded)
        if text and len(text.strip()) > 50:
            return title or _title_from_filename(filename), _normalize(text)

    # Path 2: SharePoint HTML (browser-saved, JS-rendered) — content in large script
    soup = BeautifulSoup(file_content, "lxml")
    js_text = _extract_from_js_page(soup)
    if js_text and len(js_text.strip()) > 100:
        return title or _title_from_filename(filename), _normalize(js_text)

    # Path 3: trafilatura (standard HTML articles / blogs)
    extracted = trafilatura.extract(
        file_content,
        include_tables=True,
        include_links=False,
        include_images=False,
        favor_recall=True,
    )
    if extracted and len(extracted.strip()) > 50:
        return title or _title_from_filename(filename), _normalize(extracted)

    # Path 4: last-resort BeautifulSoup body text
    body = soup.find("body") or soup
    for tag in body(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = body.get_text(separator="\n", strip=True)
    return title or _title_from_filename(filename), _normalize(text)


def _extract_from_sharepoint_html(encoded_html: str) -> str:
    """Parse the decoded inner HTML from CanvasContent1."""
    inner = BeautifulSoup(encoded_html, "lxml")

    # data-sp-rte divs contain the visible rich-text content
    rte_divs = inner.find_all("div", attrs={"data-sp-rte": True})
    if not rte_divs:
        # fallback: all text from inner HTML
        return inner.get_text(separator="\n", strip=True)

    parts = []
    for div in rte_divs:
        parts.append(_rich_text_to_str(div))
    return "\n\n".join(p for p in parts if p.strip())


def _extract_from_js_page(soup: BeautifulSoup) -> str:
    """Extract content from a JS-rendered SharePoint page (browser-saved HTML).

    The page body contains only <script> tags; the actual content is serialised
    as JSON strings inside the largest inline script bundle.  We pull out all
    Ukrainian text segments ≥ 20 chars, deduplicate in order of appearance,
    and join them as readable paragraphs.
    """
    inline_scripts = [s for s in soup.find_all("script") if s.string]
    if not inline_scripts:
        return ""

    big = max(inline_scripts, key=lambda s: len(s.string or ""))
    script_text = big.string or ""

    if len(script_text) < _JS_PAGE_SCRIPT_MIN:
        return ""  # Not a JS-rendered SharePoint page

    seen: set[str] = set()
    parts: list[str] = []

    for m in _UK_SEGMENT_RE.finditer(script_text):
        segment = re.sub(r"\s+", " ", m.group(0)).strip()
        # Drop the trailing SharePoint metadata JSON if it bled into this segment
        segment = _JSON_TAIL_RE.split(segment, 1)[0].rstrip(' ",').strip()
        # A short remainder after cutting the JSON tail is metadata residue, not
        # content (e.g. the page-editor name that was the layoutsUrl value). The
        # source regex already required ≥20 chars, so this only drops residue.
        if len(segment) < 20 or segment in seen:
            continue
        # Skip obvious code/URL fragments
        if any(kw in segment for kw in ("function(", "var ", "return ", "typeof ", "http")):
            continue
        seen.add(segment)
        parts.append(segment)

    return "\n\n".join(parts)


def _rich_text_to_str(element) -> str:
    """Recursively convert rich-text element to readable plain text.

    h1–h3 → section markers (## prefix) so the splitter can use them as
    natural break points.  h4–h6 are FAQ-style sub-headings; we keep them
    inline with their following answer text by using a Q:/A: style so the
    splitter does NOT break between question and answer.
    """
    parts = []
    children = list(element.children)
    i = 0
    while i < len(children):
        child = children[i]
        if not hasattr(child, "name"):
            text = str(child).strip()
            if text and text != "\xa0":
                parts.append(text)
            i += 1
            continue

        tag = child.name
        if tag in ("h1", "h2", "h3"):
            t = child.get_text(" ", strip=True)
            if t:
                parts.append(f"\n## {t}\n")
        elif tag in ("h4", "h5", "h6"):
            # Collect question + the next sibling paragraph(s) as one block,
            # so the splitter never separates a question from its answer.
            question = child.get_text(" ", strip=True)
            answer_parts = []
            j = i + 1
            while j < len(children):
                sib = children[j]
                if not hasattr(sib, "name"):
                    j += 1
                    continue
                if sib.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    break  # next heading-level question/section starts
                if _is_numbered_question(sib):
                    break  # <p><strong>N. question text</strong></p>  pattern
                t = sib.get_text(" ", strip=True)
                if t and t != "\xa0":
                    answer_parts.append(t)
                j += 1
            if answer_parts:
                # Newline prefix ensures splitter treats each QA pair as own block
                parts.append(f"\nПитання: {question}\nВідповідь: {' '.join(answer_parts)}\n")
                i = j
                continue
            elif question:
                parts.append(f"\nПитання: {question}\n")
        elif tag in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                t = li.get_text(" ", strip=True)
                if t:
                    parts.append(f"- {t}")
        elif tag == "p":
            # Handle <p><strong>N. question text</strong></p> as a FAQ question
            if _is_numbered_question(child):
                q_text = child.get_text(" ", strip=True)
                answer_parts = []
                j = i + 1
                while j < len(children):
                    sib = children[j]
                    if not hasattr(sib, "name"):
                        j += 1
                        continue
                    if sib.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                        break
                    if _is_numbered_question(sib):
                        break
                    t = sib.get_text(" ", strip=True)
                    if t and t != "\xa0":
                        answer_parts.append(t)
                    j += 1
                if answer_parts:
                    parts.append(f"\nПитання: {q_text}\nВідповідь: {' '.join(answer_parts)}\n")
                    i = j
                    continue
                else:
                    parts.append(f"\nПитання: {q_text}\n")
            else:
                t = child.get_text(" ", strip=True)
                if t and t != "\xa0":
                    parts.append(t)
        elif tag == "table":
            parts.append(_table_to_str(child))
        elif tag in ("div", "section", "article"):
            parts.append(_rich_text_to_str(child))
        else:
            t = child.get_text(" ", strip=True)
            if t:
                parts.append(t)
        i += 1
    return "\n".join(parts)


def _is_numbered_question(tag) -> bool:
    """True when tag is <p><strong>N. text</strong></p> — a numbered FAQ question."""
    if not hasattr(tag, "name") or tag.name != "p":
        return False
    strong = tag.find("strong")
    if not strong:
        return False
    text = strong.get_text(strip=True)
    return bool(_NUMBERED_Q_RE.match(text))


def _table_to_str(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _normalize(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_filename(filename: str) -> str:
    from pathlib import Path
    return Path(filename).stem.replace("-", " ").replace("_", " ")
