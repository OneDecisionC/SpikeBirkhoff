import threading
import time

import pytest
import launcher

from launcher import dispatch, gpu_indices, plan_jobs, command_for


PROTOCOL = {"methods": ["S", "B", "I", "A0"], "seeds": [42, 43, 44, 45, 46]}


def test_full_matrix_and_smoke_modes(tmp_path):
    jobs = plan_jobs(PROTOCOL, "train", "matrix")
    assert len(jobs) == len({job["name"] for job in jobs}) == 20
    assert len(plan_jobs(PROTOCOL, "smoke", "methods")) == 4
    assert "--smoke" in command_for(jobs[0], tmp_path, tmp_path, True)
    assert "--smoke" not in command_for(jobs[0], tmp_path, tmp_path, False)


def test_eight_gpu_dispatch_exactly_once_and_exclusive():
    active, observed, seen = set(), set(), []
    guard = threading.Lock()
    barrier = threading.Barrier(8)

    def runner(job, gpu):
        with guard:
            assert gpu not in active
            first = gpu not in observed
            active.add(gpu)
            observed.add(gpu)
        if first:
            barrier.wait(timeout=5)
        time.sleep(0.01)
        with guard:
            active.remove(gpu)
            seen.append(job["name"])
        return job

    jobs = plan_jobs(PROTOCOL, "train", "matrix")
    result = dispatch(jobs, list(range(8)), runner)
    assert len(result) == len(set(seen)) == 20
    assert observed == set(range(8))
    assert not active


def test_failure_stops_new_jobs():
    seen = []

    def runner(job, gpu):
        seen.append(job["name"])
        raise RuntimeError("intentional test failure")

    with pytest.raises(RuntimeError, match="intentional"):
        dispatch(plan_jobs(PROTOCOL, "train", "matrix"), [0], runner)
    assert len(seen) == 1


@pytest.mark.parametrize("value", ["0,0", "-1", "invalid", ""])
def test_invalid_gpu_list(value):
    with pytest.raises(ValueError):
        gpu_indices(value)


def test_transient_gpu_utilization_waits(monkeypatch):
    attempts = []

    def probe(gpu):
        attempts.append(gpu)
        if len(attempts) < 3:
            raise RuntimeError("busy")

    monkeypatch.setattr(launcher, "assert_idle", probe)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    launcher.wait_idle(0)
    assert attempts == [0, 0, 0]


def test_busy_gpu_timeout(monkeypatch):
    def probe(gpu):
        raise RuntimeError("busy")

    monkeypatch.setattr(launcher, "assert_idle", probe)
    with pytest.raises(RuntimeError, match="busy"):
        launcher.wait_idle(0, timeout=0)
