from app.image2text import _extract_text_from_responses_payload, _guess_mime_type


def test_guess_mime_type_by_extension():
    assert _guess_mime_type("a.png") == "image/png"
    assert _guess_mime_type("a.webp") == "image/webp"
    assert _guess_mime_type("a.gif") == "image/gif"
    assert _guess_mime_type("a.jpg") == "image/jpeg"


def test_extract_text_prefers_output_text_field():
    payload = {"output_text": "  hello world  "}
    assert _extract_text_from_responses_payload(payload) == "hello world"


def test_extract_text_from_nested_output_blocks():
    payload = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ]
            }
        ]
    }
    assert _extract_text_from_responses_payload(payload) == "line1\nline2"


def test_extract_text_returns_empty_for_unrecognized_payload():
    assert _extract_text_from_responses_payload({"output": []}) == ""
