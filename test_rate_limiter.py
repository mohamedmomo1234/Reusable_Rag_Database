from rate_limiter import allow_request


def test_rate_limiter():
    client = "test-client"

    results = [
        allow_request(client)
        for _ in range(20)
    ]

    assert any(result is False for result in results)
