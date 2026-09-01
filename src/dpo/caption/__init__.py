"""Caption machinery, kept apart from the instrument that drives it.

What a caption request *is*, how prose is written from one, that identical
settings return the identical caption, and how a shot's media is cut: none of
that depends on the surface a participant touched to reach the settings. So it
lives here rather than inside ``dpo.session``, which owns the surface — the
skeleton, its orderings, its copy — and builds a ``CaptionRequest`` from it.

``CaptionRequest`` is the seam. Upstream of it is one instrument's idea of what
a control is; downstream is the model, the budget, and the cache, and nothing
downstream needs to know which gesture produced the settings.

Nothing in this package may import from ``dpo.session``. The dependency runs
one way.
"""

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
