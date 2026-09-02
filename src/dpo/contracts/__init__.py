"""Caption contracts and the executable study contract.

`study_contract` owns the shared normative vocabularies (tracks, splits,
annotation choices, pair categories, the nine-condition experiment matrix) and
the study-contract validator. `captions` holds the word, length and sentence
measurements both tracks share; `visual_caption` and `audio_caption` own the
per-track contracts on top of them: deterministic compliance checks and
cross-modal lexical screens.

Every consumer imports the module it needs directly, which is why this file
re-exports nothing: a name reachable two ways is a name that can be imported
from the wrong one.
"""
