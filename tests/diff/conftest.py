"""Fixtures for diff's tests (helpers live in diff_fixtures.py)."""

from __future__ import annotations

import pytest
from diff_fixtures import relabelled, smoke_document

from diff.caption_sources import SourceSet, parse_sources


@pytest.fixture
def smoke_sources() -> SourceSet:
    return parse_sources(smoke_document())


@pytest.fixture
def development_sources() -> SourceSet:
    return parse_sources(relabelled("development"))
