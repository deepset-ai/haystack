from contextlib import contextmanager
from threading import Semaphore

from fastapi import HTTPException


class RequestLimiter:
    def __init__(self, limit):
        self.semaphore = Semaphore(limit - 1)

    @contextmanager
    def run(self):
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=503, detail="The server is busy processing requests.")
        try:
            yield acquired
        finally:
            self.semaphore.release()
