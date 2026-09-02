"""Caption machinery shared by both participant instruments.

Two instruments are built against two specifications:

* ``dpo.session`` — the skeleton-only instrument of ``docs/v1-session/``, where the
  ordered list of what a caption will mention is the sole control surface;
* ``dpo.console`` — the console instrument of ``docs/v2-console/``, where a token
  strip, a segmented crossfader and four detents drive a read-only skeleton.

They disagree about surfaces, identity, copy, screens, and how a source's
weights are computed. They agree about everything downstream of the settings:
what a caption request *is*, how prose is written from one, that identical
settings return the identical caption, and how a shot's media is cut. That
agreement lives here so the two never drift apart on the expensive, model-
shaped half, and so archiving either instrument means deleting its own
package and nothing else.

Nothing in this package may import from ``dpo.session`` or ``dpo.console``.
The dependency runs one way.
"""

from dpo.caption.background import BackgroundWriter
from dpo.caption.writer import (
    CAPTION_MAX_CHARS,
    CachedWriter,
    CaptionRequest,
    CaptionWriter,
    GemmaWriter,
    HeadSpec,
    ShotMedia,
    SourceSpec,
    TemplateWriter,
    WriterError,
)

__all__ = [
    "BackgroundWriter",
    "CAPTION_MAX_CHARS",
    "CachedWriter",
    "CaptionRequest",
    "CaptionWriter",
    "GemmaWriter",
    "HeadSpec",
    "ShotMedia",
    "SourceSpec",
    "TemplateWriter",
    "WriterError",
]
