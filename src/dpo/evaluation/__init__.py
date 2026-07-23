"""Common validation applied identically to every experiment in the matrix.

One generation adapter, one automatic metric runner, one report schema.
Reference-free similarity metrics that need external models (CLIPScore, FENSE,
CLAP) are typed external boundaries — they report
``blocked_pending_external_operation`` rather than silently substituting a
proxy — and human preference remains the primary criterion.
"""
