"""Pydantic Annotated identifier variants for wire contracts."""

from typing import Annotated

from pydantic import Field

ID_PATTERN = r"^[0-9a-f][0-9a-f-]{34}[0-9a-f]$"

SessionId = Annotated[
    str, Field(pattern=ID_PATTERN, description="Session identifier (36-char lowercase uuid4)")
]
ModelId = Annotated[
    str, Field(pattern=ID_PATTERN, description="Model identifier (36-char lowercase uuid4)")
]
RouteId = Annotated[
    str, Field(pattern=ID_PATTERN, description="Route identifier (36-char lowercase uuid4)")
]
