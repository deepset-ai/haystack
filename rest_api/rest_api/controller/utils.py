from typing import Type, NewType

import inspect
from contextlib import contextmanager
from threading import Semaphore

from fastapi import Form, HTTPException
from pydantic import BaseModel


class RequestLimiter:
    def __init__(self, limit):
        self.semaphore = Semaphore(limit)

    @contextmanager
    def run(self):
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=503, detail="The server is busy processing requests.")
        try:
            yield acquired
        finally:
            self.semaphore.release()


StringId = NewType("StringId", str)


def as_form(cls: Type[BaseModel]):
    """
    Adds an as_form class method to decorated models. The as_form class method
    can be used with FastAPI endpoints
    """
    new_params = [
        inspect.Parameter(
            field.alias,
            inspect.Parameter.POSITIONAL_ONLY,
            default=(Form(field.default) if not field.required else Form(...)),
        )
        for field in cls.__fields__.values()
    ]

    async def _as_form(**data):
        return cls(**data)

    sig = inspect.signature(_as_form)
    sig = sig.replace(parameters=new_params)
    _as_form.__signature__ = sig  # type: ignore
    setattr(cls, "as_form", _as_form)
    return cls
