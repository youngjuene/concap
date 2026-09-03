SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: sync check test lint typecheck locks smoke canary golden annotate report session-demo console-demo regen-demo

# Operator loop for the live study (configs/study/street-audio.toml).
SPLIT ?= train
annotate:
	uv run dpo annotation serve --tasks data/annotation/tasks-$(SPLIT).json \
	  --media-dir data/live/media --out data/annotation/responses

report:
	uv run dpo report show --workspace artifacts/street

# The caption session over synthetic clips (docs/v1-session/runbook.md): stage
# demo media for the fixture document, then serve it with the template writer.
session-demo:
	uv run python scripts/stage_session_demo.py \
	  --session tests/session/fixtures/session.json --out data/session-demo/media
	uv run dpo session serve --session tests/session/fixtures/session.json \
	  --media-dir data/session-demo/media --out data/session-demo/responses --writer template

# The console instrument (docs/v2-console/runbook.md) over its own document and
# its own staging of the same synthetic clip. Serves on 8778, one past the
# session's 8777, so both can run at once for a side-by-side comparison.
console-demo:
	uv run python scripts/stage_session_demo.py \
	  --session tests/console/fixtures/console.json --out data/console-demo/media
	uv run dpo console serve --session tests/console/fixtures/console.json \
	  --media-dir data/console-demo/media --out data/console-demo/responses --writer template

# The regeneration instrument (docs/v3-regen/runbook.md) over its own synthetic
# staging: two segments, masks, stems whose envelopes match the document's
# waveforms, and the placeholder item set. Serves on 8779, one past the
# console's 8778, so all three can run at once.
regen-demo:
	uv run python scripts/stage_regen_demo.py \
	  --session tests/regen/fixtures/regen.json --out data/regen-demo/media
	uv run dpo regen serve --session tests/regen/fixtures/regen.json \
	  --media-dir data/regen-demo/media --out data/regen-demo/responses --writer template

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
