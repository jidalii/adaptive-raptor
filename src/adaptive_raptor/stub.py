"""Marker for research components that are not implemented yet.

A placeholder class inherits from :class:`Stub`. Runners call :func:`ensure_implemented`
on every component they will use *before* any expensive work (index builds, API calls),
so an experiment that depends on a placeholder stops immediately with
``NotImplementedError`` and is reported as ``not_implemented``.

To implement a component, write the methods and remove ``Stub`` from its bases.
"""

from __future__ import annotations


class Stub:
    """Base class for placeholder research components."""


def is_stub(obj: object) -> bool:
    cls = obj if isinstance(obj, type) else type(obj)
    return issubclass(cls, Stub)


def ensure_implemented(*objs: object) -> None:
    missing = [(o if isinstance(o, type) else type(o)).__name__ for o in objs if is_stub(o)]
    if missing:
        raise NotImplementedError(f"placeholder component(s) not implemented: {', '.join(missing)}")
