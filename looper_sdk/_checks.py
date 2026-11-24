"""The checks the SDK's values make of what they are given: a whole count, a
finite number, a flag. Each raises ``InvalidArgument`` naming the field."""

from __future__ import annotations

import math
import operator

from looper_sdk.errors import InvalidArgument


def _count(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise InvalidArgument(f"{name} must be an integer, not {value!r}")
    try:
        count = operator.index(value)  # pyright: ignore[reportArgumentType]
    except TypeError:
        raise InvalidArgument(f"{name} must be an integer, not {value!r}") from None
    if count < minimum:
        raise InvalidArgument(f"{name} must be at least {minimum}, not {count}")
    return count


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool | str | bytes):
        raise InvalidArgument(f"{name} must be a number, not {value!r}")
    try:
        number = float(value)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        raise InvalidArgument(f"{name} must be a number, not {value!r}") from None
    if not math.isfinite(number):
        raise InvalidArgument(f"{name} must be finite, not {value!r}")
    return number


def _flag(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise InvalidArgument(f"{name} must be True or False, not {value!r}")
    return value
