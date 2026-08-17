"""Python 3.10+ compatibility shim for Experta's legacy dependency chain.

Experta 1.9.4 pins frozendict==1.2, which subclasses `collections.Mapping` — an
alias removed from the stdlib in Python 3.10 (moved to collections.abc). Importing
this module BEFORE `import experta` restores the aliases so the whole chain loads
cleanly on modern Python (verified on 3.13).
"""
from __future__ import annotations

import collections
import collections.abc

for _name in (
    "Mapping", "MutableMapping", "Sequence", "MutableSequence",
    "Iterable", "Iterator", "Callable", "Hashable", "Set", "MutableSet",
):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
