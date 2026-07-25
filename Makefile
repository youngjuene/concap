SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: sync check test lint typecheck locks smoke canary live-boundary-smoke golden

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

smoke: canary live-boundary-smoke

canary:
	@tmp="$$(mktemp -d)"; \
	cold="$$tmp/runs/canary-cold.json"; warm="$$tmp/runs/canary-warm.json"; \
	mkdir -p "$$tmp/runs"; \
	uv run dpo canary run --workspace "$$tmp/artifacts" --contract configs/study/canary.toml > "$$cold"; \
	uv run dpo canary run --workspace "$$tmp/artifacts" --contract configs/study/canary.toml > "$$warm"; \
	uv run dpo artifact verify --workspace "$$tmp/artifacts" --all >/dev/null; \
	uv run python -c 'import json,sys; cold=json.load(open(sys.argv[1])); warm=json.load(open(sys.argv[2])); assert cold["artifact_id"] == warm["artifact_id"], (cold["artifact_id"], warm["artifact_id"]); assert warm["cached"] is True, warm; assert warm["provider_calls"] == 0, warm; assert cold["status"] == "offline_milestone_complete", cold; print("canary ok", cold["artifact_id"])' "$$cold" "$$warm"

live-boundary-smoke:
	@tmp="$$(mktemp -d)"; out="$$tmp/live-boundary.json"; \
	set +e; \
	uv run dpo evaluate run --workspace "$$tmp/artifacts" --contract configs/study/canary.toml \
	  --track visual --invoke-external > "$$out"; status="$$?"; \
	set -e; \
	[ "$$status" -eq 3 ]; \
	uv run python -c 'import json,sys; doc=json.load(open(sys.argv[1])); assert doc["status"] == "blocked_pending_external_operation", doc; assert doc["side_effects"] is False, doc; print("live boundary ok", doc["command"])' "$$out"
