from enfold.llm_extract import _parse_response


def test_parse_valid_array():
    raw = '[{"content": "The user prefers dark mode in the editor.", "category": "user_pref", "tags": "ui,editor"}]'
    out = _parse_response(raw)
    assert len(out) == 1
    assert out[0]["category"] == "user_pref"
    assert out[0]["tags"] == "ui,editor"


def test_parse_strips_markdown_fences():
    raw = '```json\n[{"content": "The project deploys from the main branch.", "category": "project", "tags": "ci,deploy"}]\n```'
    assert len(_parse_response(raw)) == 1


def test_parse_empty_and_garbage_are_safe():
    assert _parse_response("[]") == []
    assert _parse_response("not json at all") == []
    assert _parse_response('{"not": "a list"}') == []
    assert _parse_response("") == []


def test_parse_validates_category_and_min_length():
    raw = (
        '[{"content": "short", "category": "user_pref", "tags": ""},'
        ' {"content": "A sufficiently long and durable fact statement.", "category": "bogus", "tags": "x"}]'
    )
    out = _parse_response(raw)
    # "short" (<10 chars) dropped; bad category coerced to "general"
    assert len(out) == 1
    assert out[0]["category"] == "general"


def test_parse_truncates_long_content():
    long = "x" * 800
    raw = '[{"content": "' + long + '", "category": "general", "tags": ""}]'
    out = _parse_response(raw)
    assert len(out) == 1
    assert len(out[0]["content"]) <= 400
