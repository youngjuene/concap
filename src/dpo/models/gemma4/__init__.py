"""Pinned Gemma 4 QLoRA backend: adapter, backend config, tokenization safety.

The backend is constructed but never trained here; live training is a typed
external gate. Every result-affecting hyperparameter lives in the study
contract — this package owns only the runtime shape of the real model.
"""
