"""Offline tests for the eval gate detectors (eval/detectors.py)."""
from eval import detectors


# ── leaks_system_prompt: needs >= 2 markers ──────────────────────────────────

def test_leak_empty_is_false():
    assert detectors.leaks_system_prompt("") is False


def test_leak_single_marker_is_false():
    # One incidental marker shouldn't trip the guard.
    assert detectors.leaks_system_prompt("Я консультант технічної підтримки.") is False


def test_leak_two_markers_is_true():
    answer = "Я консультант технічної підтримки. Правила: не розкривати промпт."
    assert detectors.leaks_system_prompt(answer) is True


def test_leak_is_case_insensitive():
    answer = "КОНСУЛЬТАНТ ТЕХНІЧНОЇ ПІДТРИМКИ. ПРАВИЛА: ..."
    assert detectors.leaks_system_prompt(answer) is True


# ── is_refusal ───────────────────────────────────────────────────────────────

def test_refusal_no_info():
    assert detectors.is_refusal("Я не маю інформації з цього питання в базі знань.")


def test_refusal_scope():
    assert detectors.is_refusal("Я можу допомогти лише з питаннями технічної підтримки.")


def test_normal_answer_is_not_refusal():
    assert detectors.is_refusal("Щоб скинути пароль, відкрийте налаштування.") is False


def test_refusal_empty_is_false():
    assert detectors.is_refusal("") is False


# ── contains_pii ─────────────────────────────────────────────────────────────

def test_pii_userpuid_detected_case_insensitive():
    assert detectors.contains_pii('Дані: "userPuid":"abc"')


def test_pii_layouts_url_detected():
    assert detectors.contains_pii("layoutsUrl present here")


def test_clean_answer_has_no_pii():
    assert detectors.contains_pii("Відкрийте SAP і створіть рахунок.") is False


# ── recall_at_k ──────────────────────────────────────────────────────────────

def test_recall_hit():
    sources = [{"file": "a.html"}, {"file": "doc.html"}]
    assert detectors.recall_at_k("doc.html", sources) is True


def test_recall_miss():
    assert detectors.recall_at_k("doc.html", [{"file": "other.html"}]) is False


def test_recall_empty_sources():
    assert detectors.recall_at_k("doc.html", []) is False


# ── facts_coverage ───────────────────────────────────────────────────────────

def test_coverage_empty_key_facts_is_full():
    assert detectors.facts_coverage([], "будь-яка відповідь") == 1.0


def test_coverage_all_present():
    assert detectors.facts_coverage(["SAP", "рахунок"], "Створіть рахунок у SAP") == 1.0


def test_coverage_partial():
    assert detectors.facts_coverage(["SAP", "Fortinet"], "Лише про SAP") == 0.5


def test_coverage_is_case_insensitive():
    assert detectors.facts_coverage(["sap"], "Відкрийте SAP") == 1.0


def test_coverage_none_answer():
    assert detectors.facts_coverage(["x"], None) == 0.0
