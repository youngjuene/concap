"""Content-addressed artifact infrastructure shared by every pipeline stage.

`identity` defines canonical semantic encoding and hashing; `atomic` publishes
immutable bytes; `safety` owns destructive-path and checkpoint-deserialization
guards; `access` is the fenced capability authority for protected test/study
payloads; `artifacts` is the content-addressed
artifact registry with recursive lineage, role-exposure, and leakage checks.
"""
