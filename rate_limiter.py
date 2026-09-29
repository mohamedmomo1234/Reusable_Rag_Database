import time
from collections import defaultdict, deque

from config import MAX_REQUESTS_PER_MINUTE

_requests = defaultdict(deque)


def allow_request(client_id):
    now = time.time()
    window = 60

    history = _requests[client_id]

    while history and now - history[0] >= window:
        history.popleft()

    if len(history) >= MAX_REQUESTS_PER_MINUTE:
        return False

    history.append(now)
    return True
