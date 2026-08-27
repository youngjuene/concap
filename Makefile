SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: sync check test lint typecheck locks smoke canary golden annotate report

# Operator loop for the live study (configs/study/street-audio.toml).
SPLIT ?= train
annotate:
	uv run dpo annotation serve --tasks data/annotation/tasks-$(SPLIT).json \
	  --media-dir data/live/media --out data/annotation/responses

report:
	uv run dpo report show --workspace artifacts/street

sync:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:
	uv run mypy

locks:
	uv lock --check

golden:
	uv run python scripts/regenerate_golden.py

check: lint typecheck test locks

smoke: canary

canary:
	@tmp="$$(mktemp -d)"; \
	cold="$$tmp/runs/canary-cold.json"; warm="$$tmp/runs/canary-warm.json"; \
	mkdir -p "$$tmp/runs"; \
	uv run dpo canary run --workspace "$$tmp/artifacts" --contract configs/study/canary.toml > "$$cold"; \
	uv run dpo canary run --workspace "$$tmp/artifacts" --contract configs/study/canary.toml > "$$warm"; \
	uv run dpo artifact verify --workspace "$$tmp/artifacts" --all >/dev/null; \
	uv run python -c 'import json,sys; cold=json.load(open(sys.argv[1])); warm=json.load(open(sys.argv[2])); assert cold["artifact_id"] == warm["artifact_id"], (cold["artifact_id"], warm["artifact_id"]); assert warm["cached"] is True, warm; assert warm["provider_calls"] == 0, warm; assert cold["status"] == "offline_milestone_complete", cold; print("canary ok", cold["artifact_id"])' "$$cold" "$$warm"
