"""Audit public browser exports without trusting the browser's pass/fail summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dpo.regen.study_schema import digest, load_manifest


def verify(folder: Path, manifest_path: Path, media: Path) -> dict[str, object]:
    manifest = load_manifest(manifest_path, media)
    journeys = []
    for path in sorted(folder.glob("*-export.json")):
        exported = json.loads(path.read_text())
        state = exported["state"]
        assert state["stage"] == "done"
        assert state["viewing"] == manifest["viewing_videos"]
        assert len(state["completions"]) == 3
        source = state["calibration_source"]
        assert source["participant"].startswith("qa-public-")
        assert len(source["viewings"]) == 2
        assert sorted(len(s["responses"]) for s in source["responses"]) == [8, 22]
        assert state["profile"]["hash"] == digest(
            {key: value for key, value in state["profile"].items() if key != "hash"}
        )
        survey = state["final_survey"]
        assert survey["profile_hash"] == state["profile"]["hash"]
        assert survey["viewing_hash"] == state["viewing_hash"]
        assert survey["instrument"] == state["instrument"]
        assert survey["items_hash"] == digest(state["instrument"]["items"]["final"])
        actions = [
            (event["at"], json.loads(event["body"]))
            for event in exported["events"]
            if event["body"].startswith("{")
        ]
        final = [at for at, body in actions if body.get("action") == "final-survey"]
        start = [at for at, body in actions if body.get("action") == "start-viewing"]
        assert len(final) == len(start) == 1
        assert final[0] - start[0] >= sum(v["duration_ms"] for v in state["viewing"]) / 1000 - 3
        jobs = {job["id"]: job for job in exported["jobs"]}
        exposures = exported["exposures"]
        assert len({entry["id"] for entry in exposures}) == len(exposures)
        videos = []
        for video, completion in zip(state["viewing"], state["completions"], strict=True):
            assert completion["video_id"] == video["id"]
            coverage = sum(end - begin for begin, end in completion["coverage"])
            assert coverage >= video["duration_ms"] - 1500
            entries = [entry for entry in exposures if entry["video_id"] == video["id"]]
            assert {entry["cue"] for entry in entries} == set(range(len(video["cues"])))
            generated = [entry for entry in entries if not entry["fallback"]]
            for level in (0, 1):
                assert any(e["axes"] == {"texture": level, "context": level} for e in generated)
            for entry in entries:
                cue = video["cues"][entry["cue"]]
                assert cue["start_ms"] <= entry["start_ms"] <= entry["end_ms"] <= cue["end_ms"]
                if entry["fallback"]:
                    assert entry["text"] == cue["fallback"][state["language"]]
                else:
                    job = jobs[entry["job_id"]]
                    result, spec = json.loads(job["result"]), json.loads(job["spec"])
                    assert spec["profile_hash"] == state["profile"]["hash"]
                    assert spec["model_identity"]["settings"].get("backend_config")
                    assert result["fallback"] is False and entry["text"] == result["text"]
                    assert entry["axes"] == spec["axes"]
                    assert entry["settings_revision"] == job["revision"]
                    assert spec["start_ms"] == cue["start_ms"] and spec["end_ms"] == cue["end_ms"]
                    assert Path(spec["audio"]).resolve() == (media / video["audio"]).resolve()
            videos.append(
                {
                    "id": video["id"],
                    "coverage_ms": coverage,
                    "cue_windows": len(video["cues"]),
                    "generated_exposures": len(generated),
                    "fallback_exposures": len(entries) - len(generated),
                    "brief_and_detailed_verified": True,
                }
            )
        journeys.append(
            {
                "participant": source["participant"],
                "stage": state["stage"],
                "language": state["language"],
                "videos": videos,
                "final_surveys": len(final),
                "elapsed_seconds_including_preparation": final[0] - start[0],
            }
        )
    assert journeys, "No completed browser export to verify"
    return {"passed": True, "manifest_hash": digest(manifest), "journeys": journeys}


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    result = verify(folder, Path(sys.argv[2]), Path(sys.argv[3]))
    (folder / "record-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
