"""Independent single-GPU jobs; explicit smoke mode and frozen validation selection."""
import argparse
import concurrent.futures
import contextlib
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import shutil
import sys
import threading
import time

from common import (ROOT, assert_idle, check_data, code_hashes, execute, expected_jobs,
                    host_identity, read_json, run_name, validate_training, verify_freeze, write_new)
from rental_data import partition_records, sha256, verify_split


def gpu_indices(value):
    result = [int(item) for item in value.split(",")]
    if not result or min(result) < 0 or len(result) != len(set(result)):
        raise ValueError("GPU indices must be unique and nonnegative")
    return result


def plan_jobs(protocol, mode, smoke_scope):
    jobs = expected_jobs(protocol)
    if mode == "smoke" and smoke_scope == "methods":
        jobs = [(method, protocol["seeds"][0]) for method in protocol["methods"]]
    return [{"name": run_name(method, seed), "method": method, "seed": seed}
            for method, seed in jobs]


def dispatch(jobs, gpus, runner):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    stop = threading.Event()
    results, errors = [], []
    guard = threading.Lock()

    def lane(gpu):
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            if stop.is_set():
                return
            try:
                result = runner(job, gpu)
                with guard:
                    results.append(result)
            except Exception as error:
                with guard:
                    errors.append((job["name"], str(error)))
                stop.set()
                return

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lane, gpus))
    if errors:
        raise RuntimeError("Queue stopped; already-running jobs were allowed to finish: " + repr(errors))
    if len(results) != len(jobs):
        raise RuntimeError("Incomplete queue")
    return sorted(results, key=lambda row: row["name"])


@contextlib.contextmanager
def acquire_lock(path):
    import fcntl
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "numpy", "timm", "spikingjelly", "cupy-cuda11x", "tensorboard")}


def initialize(output, data, split_path, mode, scope, jobs):
    protocol = read_json(ROOT / "protocol.json")
    split = read_json(split_path)
    verify_split(split)
    records = [row for group in split["partitions"].values() for row in group]
    if split["seed"] != protocol["split_seed"] or partition_records(records, split["seed"]) != split["partitions"]:
        raise ValueError("Split is not the prescribed fixed 8:1:1 partition")
    check_data(data, split)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("runs", "logs", "receipts"):
        (output / name).mkdir()
    shutil.copyfile(split_path, output / "split.json")
    write_new(output / "freeze.json", {"code_sha256": code_hashes(),
              "split_sha256": sha256(output / "split.json"), "data_absolute_path": str(data),
              "output_absolute_path": str(output), "mode": mode, "smoke_scope": scope,
              "jobs": jobs, "versions": versions(), "server": host_identity()})


def command_for(job, output, data, smoke):
    command = [sys.executable, "-m", "spikebirkhoff.train", "--config", str(ROOT / "config.yaml"),
               "--method", job["method"], "--data-dir", str(data), "--output", str(output / "runs"),
               "--experiment", job["name"], "--set", "seed=" + str(job["seed"]),
               "--set", "split_manifest=" + str(output / "split.json")]
    if smoke:
        command.append("--smoke")
    return command


def completed_record(job, output, smoke, epochs):
    run = output / "runs" / job["name"]
    meta = read_json(run / "run.json")
    if meta["status"] != "finished" or meta["smoke_test"] != smoke or meta["seed"] != job["seed"]:
        raise ValueError("Incomplete or wrong-mode run")
    if not smoke:
        return validate_training(run, epochs)
    with (run / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or int(rows[0]["epoch"]) != 0:
        raise ValueError("Invalid smoke epoch count")
    for key in ("train_loss", "eval_loss", "eval_top1"):
        if not math.isfinite(float(rows[0][key])):
            raise ValueError("Non-finite smoke metric")
    return {"checkpoint": str(run / "model_best.pth.tar"),
            "checkpoint_sha256": sha256(run / "model_best.pth.tar"),
            "summary_sha256": sha256(run / "summary.csv"),
            "resolved_sha256": sha256(run / "resolved.yaml")}


def wait_idle(gpu, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        try:
            assert_idle(gpu)
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def run_job(job, gpu, output, freeze, epochs):
    receipt = output / "receipts" / (job["name"] + ".done.json")
    smoke = freeze["mode"] == "smoke"
    if receipt.exists():
        record = read_json(receipt)
        actual = completed_record(job, output, smoke, epochs)
        if any(record.get(key) != value for key, value in dict(actual, **job).items()):
            raise ValueError("Completed job changed")
        return record
    started = output / "receipts" / (job["name"] + ".started.json")
    if started.exists() or (output / "runs" / job["name"]).exists():
        raise ValueError("Interrupted job: no automatic restart or checkpoint resume")
    command = command_for(job, output, freeze["data_absolute_path"], smoke)
    wait_idle(gpu)
    write_new(started, {"command": command, "gpu": gpu, "server": host_identity()})
    elapsed = execute(command, gpu, output / "logs" / (job["name"] + ".log"))
    record = dict(completed_record(job, output, smoke, epochs), **job,
                  gpu=gpu, elapsed_seconds=elapsed, server=host_identity(),
                  run_absolute_path=str(output / "runs" / job["name"]))
    write_new(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--smoke-scope", choices=("methods", "matrix"), default="matrix")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = gpu_indices(args.gpus)
    protocol = read_json(ROOT / "protocol.json")
    jobs = plan_jobs(protocol, args.mode, args.smoke_scope)
    if args.dry_run:
        print(json.dumps({"gpus": gpus, "mode": args.mode, "jobs": jobs,
                          "internal_test_evaluated": False}, indent=2))
        return
    if args.data is None or args.split is None or args.output is None:
        parser.error("--data, --split and --output are required")
    output, data, split_path = args.output.resolve(), args.data.resolve(), args.split.resolve()
    if output == ROOT or ROOT in output.parents or data == output or data in output.parents:
        parser.error("Results must be outside the code and dataset directories")
    if args.mode == "train" and args.smoke_scope != "matrix":
        parser.error("Training requires the full matrix")
    with contextlib.ExitStack() as stack:
        output.parent.mkdir(parents=True, exist_ok=True)
        stack.enter_context(acquire_lock(output.parent / (output.name + ".lock")))
        for gpu in gpus:
            stack.enter_context(acquire_lock(Path("/tmp") / ("cifar10dvs_gpu_{}.lock".format(gpu))))
            assert_idle(gpu)
        if not args.resume:
            initialize(output, data, split_path, args.mode, args.smoke_scope, jobs)
        freeze = verify_freeze(output)
        if (freeze["mode"] != args.mode or freeze["jobs"] != jobs or freeze["versions"] != versions()
                or freeze["data_absolute_path"] != str(data) or freeze["split_sha256"] != sha256(split_path)):
            raise ValueError("Resume inputs differ from the frozen experiment")
        if args.resume:
            check_data(data, read_json(output / "split.json"))
        results = dispatch(jobs, gpus, lambda job, gpu: run_job(job, gpu, output, freeze, protocol["epochs"]))
        verify_freeze(output)
        if args.mode == "train":
            destination = output / "selection_frozen_before_test.json"
            if destination.exists():
                if read_json(destination) != results:
                    raise ValueError("Frozen selections changed")
            else:
                write_new(destination, results)
        marker = output / ("SMOKE_COMPLETE.json" if args.mode == "smoke" else "TRAINING_COMPLETE.json")
        if not marker.exists():
            write_new(marker, {"mode": args.mode, "per_run": results, "internal_test_evaluated": False,
                               "next_step": "Controlled test evaluation requires a separate command; this launcher never evaluates test data."})
        print("Completed: " + str(marker), flush=True)


if __name__ == "__main__":
    main()
