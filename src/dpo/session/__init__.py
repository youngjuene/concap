"""The caption session: a participant shapes a clip's captions by working a list.

The instrument in ``docs/v1-session/spec-behavior.md``. A
participant watches a street-scene clip, lists the sounds they noticed, shapes
each shot's caption through the skeleton (admission, balance, detail — spec
Table 2), watches the clip again with their captions placed, and answers a few
items; on a later day they complete a follow-up in a browser.

Deliberately separate from ``dpo.userstudy`` (a congruency slider over a
published ladder) and ``dpo.annotation`` (pairwise preferences from experts):
this instrument's inputs are an authored session document, its responses are an
event log per participant, and none of it is a pipeline artifact. Sharing a
schema with either would let one study's data satisfy another's validator.

Throughout this package a bare ``spec N`` cites
``docs/v1-session/spec-behavior.md`` and ``spec-identity.md ...`` cites the
identity beside it. ``media`` now lives in ``dpo.caption``, shared with the
console instrument.

Modules: ``document`` (schema and participant narrowing), ``skeleton`` (the
orderings math), ``writer`` (caption requests from settings), ``log`` (the
append-only event log), ``app`` (the HTTP surface). The html/css/js beside them
are package data read through ``importlib.resources``.
"""
