"""The console instrument of ``docs/v2-console/``.

A participant watches a clip, shapes each shot's caption with three
controls — which sources the caption mentions, how the balance between what is
seen and what is heard ranks them, and how much detail it goes into — watches
it again with the captions kept, and chooses between their caption and the
default policy's.

Throughout this package a bare ``§N`` cites ``docs/v2-console/spec-system.md``
or ``spec-uiux.md``; every module says which at the end of its docstring,
because the two number their sections independently.

Separate from :mod:`dpo.session`, which builds the skeleton instrument of
``docs/v1-session/`` against a specification this one contradicts on the control
surface, the identity, the screens and the copy. The two share
:mod:`dpo.caption` and nothing else, so whichever is not adopted can be deleted
whole.
"""

from dpo.console.config import Calibration, Configuration
from dpo.console.document import (
    CONSOLE_SCHEMA,
    ConsoleDocumentError,
    load_console_document,
    validate_console_document,
)
from dpo.console.quantities import Field, Source

__all__ = [
    "CONSOLE_SCHEMA",
    "Calibration",
    "Configuration",
    "ConsoleDocumentError",
    "Field",
    "Source",
    "load_console_document",
    "validate_console_document",
]
