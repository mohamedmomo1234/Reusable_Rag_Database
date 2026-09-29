from output_guard import sanitize_output


def test_secret_redaction():
    text = "secret gsk_1234567890ABC"
    result = sanitize_output(text)
    assert "gsk_1234567890ABC" not in result
    assert "[REDACTED]" in result
