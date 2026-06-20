"""Offline tests for the HTML/ASPX extraction pipeline (app/cleaner.py).

Fixtures are tiny synthetic documents, not the 1 MB production files, so the
regex-heavy paths are exercised deterministically and fast.
"""
import html

from bs4 import BeautifulSoup

from app import cleaner
from app.cleaner import (
    _is_numbered_question,
    _normalize,
    _table_to_str,
    _title_from_filename,
    extract_text,
)


# ── _normalize ───────────────────────────────────────────────────────────────

def test_normalize_collapses_nbsp_and_spaces():
    assert _normalize("a\xa0\xa0b   c") == "a b c"


def test_normalize_collapses_blank_lines():
    assert _normalize("a\n\n\n\n\nb") == "a\n\nb"


def test_normalize_strips_edges():
    assert _normalize("   text   ") == "text"


# ── _title_from_filename ─────────────────────────────────────────────────────

def test_title_from_filename_replaces_separators():
    assert _title_from_filename("Підтримка-касового_місця.aspx") == "Підтримка касового місця"


# ── _is_numbered_question ────────────────────────────────────────────────────

def _p(html_str):
    return BeautifulSoup(html_str, "lxml").find("p")


def test_numbered_question_true():
    assert _is_numbered_question(_p("<p><strong>5. Що робити?</strong></p>")) is True


def test_numbered_question_without_strong_is_false():
    assert _is_numbered_question(_p("<p>5. Що робити?</p>")) is False


def test_numbered_question_non_numbered_is_false():
    assert _is_numbered_question(_p("<p><strong>Загальне</strong></p>")) is False


# ── _table_to_str ────────────────────────────────────────────────────────────

def test_table_to_str_pipes_rows():
    table = BeautifulSoup(
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
        "lxml",
    ).find("table")
    assert _table_to_str(table) == "A | B\n1 | 2"


# ── Path 1: SharePoint ASPX (CanvasContent1 in IE conditional comment) ────────

def _make_aspx(inner_html: str, title: str = "Тестова інструкція") -> str:
    """Wrap rich-text inner HTML the way a real SharePoint .aspx stores it:
    HTML-entity-encoded inside an mso:CanvasContent1 element in a conditional comment."""
    encoded = html.escape(inner_html)
    return (
        f"<html><head><title>{title}</title></head><body>"
        f"<!--[if gte mso 9]><xml><mso:CanvasContent1>{encoded}"
        f"</mso:CanvasContent1></xml><![endif]--></body></html>"
    )


def test_aspx_extracts_title_and_heading():
    inner = (
        '<div data-sp-rte="true">'
        "<h2>Налаштування доступу</h2>"
        "<p>Відкрийте систему керування доступом.</p>"
        "</div>"
    )
    title, text = extract_text(_make_aspx(inner), "doc.aspx")
    assert title == "Тестова інструкція"
    assert "## Налаштування доступу" in text
    assert "Відкрийте систему керування доступом." in text


def test_aspx_groups_faq_question_with_answer():
    inner = (
        '<div data-sp-rte="true">'
        "<h4>Як скинути пароль?</h4>"
        "<p>Натисніть кнопку відновлення пароля.</p>"
        "</div>"
    )
    _, text = extract_text(_make_aspx(inner), "doc.aspx")
    # The h4 question and its answer paragraph must stay in one block.
    assert "Питання: Як скинути пароль?" in text
    assert "Відповідь: Натисніть кнопку відновлення пароля." in text


# ── Path 2: JS-rendered SharePoint page (content inside a big inline script) ──

def _make_js_page(segments: list[str], title: str = "JS сторінка") -> str:
    # Pad past the 300 KB threshold. Braces mimic the real JSON-bundle structure
    # and, crucially, break the Cyrillic-segment regex runs the way real script
    # punctuation does — so each segment is matched in isolation, not merged.
    filler = '{"pad":12345}' * 25_000  # ~325 KB
    wrapped = "".join("{" + s + "}" for s in segments)
    body = filler + wrapped + filler
    return f"<html><head><title>{title}</title></head><body><script>{body}</script></body></html>"


# Long enough on its own to clear the >100-char gate that path 2 applies.
_LONG = (
    "Ця докладна інструкція описує процес налаштування доступу до корпоративної "
    "системи керування користувачами та підтримки обладнання в компанії роздрібної торгівлі."
)


def test_js_page_extracts_and_deduplicates():
    _, text = extract_text(_make_js_page([_LONG, _LONG]), "page.html")
    assert _LONG in text
    # Duplicate appears only once.
    assert text.count("Ця докладна інструкція описує процес") == 1


def test_js_page_strips_trailing_metadata_json():
    # A content segment that bled into the SharePoint nav/auth JSON blob, alongside
    # a normal long segment so the page clears path 2's content-length gate.
    bleed = 'Автор статті Петренко Петро","userPuid":"secret-123","layoutsUrl":"x"'
    _, text = extract_text(_make_js_page([_LONG, bleed]), "page.html")
    assert _LONG in text          # extraction actually happened
    assert "userPuid" not in text
    assert "secret-123" not in text
    assert "layoutsUrl" not in text


def test_js_page_skips_code_and_url_fragments():
    code_seg = "Виклик function( щось ) повертає http://example.com результат документації тут"
    _, text = extract_text(_make_js_page([code_seg, _LONG]), "page.html")
    assert "function(" not in text
    assert "http://example.com" not in text
    assert _LONG in text


# ── Path 3/4: standard HTML article (trafilatura, or body-text fallback) ──────

def test_standard_html_extracts_body_text():
    doc = (
        "<html><head><title>Стаття</title></head><body><article>"
        "<h1>Заголовок статті</h1>"
        "<p>Цей абзац містить корисну інформацію про налаштування обладнання "
        "та підтримку користувачів у великій компанії роздрібної торгівлі.</p>"
        "</article></body></html>"
    )
    title, text = extract_text(doc, "article.html")
    assert title == "Стаття"
    assert "корисну інформацію про налаштування обладнання" in text


def test_empty_document_yields_filename_title():
    title, text = extract_text("<html><body></body></html>", "my-file.html")
    assert title == "my file"
    assert text == ""
