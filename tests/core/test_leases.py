from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dpo.core.leases import FencedLeaseError, GpuLeaseDatabase, LeaseOwner

OWNER_A = LeaseOwner(host="host", pid=10, boot_id="boot-a")
OWNER_B = LeaseOwner(host="host", pid=11, boot_id="boot-a")


def test_lease_heartbeat_release_and_monotonic_stale_recovery(tmp_path: Path) -> None:
    leases = GpuLeaseDatabase(tmp_path / "leases.sqlite")
    first = leases.acquire("GPU-0", OWNER_A, ttl_seconds=10, now=100)
    leases.heartbeat(first, OWNER_A, ttl_seconds=10, now=105)
    with pytest.raises(FencedLeaseError):
        leases.acquire("GPU-0", OWNER_B, ttl_seconds=10, now=106)
    recovered = leases.acquire("GPU-0", OWNER_B, ttl_seconds=10, now=116)
    assert recovered.fencing_token > first.fencing_token
    with pytest.raises(FencedLeaseError, match="stale"):
        leases.heartbeat(first, OWNER_A, ttl_seconds=10, now=117)
    with pytest.raises(FencedLeaseError):
        leases.release(recovered, OWNER_A)
    leases.release(recovered, OWNER_B)


def test_boot_identity_is_part_of_lease_owner(tmp_path: Path) -> None:
    leases = GpuLeaseDatabase(tmp_path / "leases.sqlite")
    token = leases.acquire("GPU-0", OWNER_A, ttl_seconds=10, now=100)
    rebooted = LeaseOwner(host="host", pid=10, boot_id="boot-b")
    with pytest.raises(FencedLeaseError):
        leases.heartbeat(token, rebooted, ttl_seconds=10, now=101)


def test_concurrent_gpu_claims_have_one_winner(tmp_path: Path) -> None:
    database = tmp_path / "leases.sqlite"

    def claim(index: int) -> int | None:
        leases = GpuLeaseDatabase(database)
        owner = LeaseOwner(host="host", pid=100 + index, boot_id="boot")
        try:
            return leases.acquire("GPU-0", owner, ttl_seconds=30, now=100).fencing_token
        except FencedLeaseError:
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(claim, range(24)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
