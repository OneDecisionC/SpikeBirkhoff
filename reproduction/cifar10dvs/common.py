"""Portable experiment bookkeeping and fail-closed integrity checks."""
import csv
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from rental_data import sha256, verify_split

ROOT = Path(__file__).resolve().parent


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hashes():
    return {path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted(ROOT.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".sh", ".yaml", ".json", ".txt", ".md")
            and not any(part.startswith(".") or part == "__pycache__"
                        for part in path.relative_to(ROOT).parts)
            and path.name not in ("MANIFEST.json", "environment-installed.txt")}


def verify_freeze(output):
    output = Path(output)
    freeze = read_json(output / "freeze.json")
    if freeze["code_sha256"] != code_hashes():
        raise ValueError("Frozen code/config changed; do not continue this experiment")
    if sha256(output / "split.json") != freeze["split_sha256"]:
        raise ValueError("Frozen split changed")
    verify_split(read_json(output / "split.json"))
    return freeze


def check_data(data, split):
    root = Path(data) / "frames_number_16_split_by_number"
    records = [row for group in split["partitions"].values() for row in group]
    actual = {path.relative_to(root).as_posix() for path in root.glob("*/*.npz")}
    if actual != {row["relative"] for row in records}:
        raise ValueError("Dataset recording list changed")
    for row in records:
        if sha256(root / row["relative"]) != row["sha256"]:
            raise ValueError("Dataset recording changed: " + row["relative"])


def expected_jobs(protocol):
    return [(method, seed) for seed in protocol["seeds"] for method in protocol["methods"]]


def run_name(method, seed):
    return "cifar10dvs_d2_t16_{}_seed{}".format(method, seed)


def environment(gpu):
    result = dict(os.environ)
    result.update(CUDA_VISIBLE_DEVICES=str(gpu),
                  PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "vendor" / "src"),
                  OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    return result


def assert_idle(gpu):
    state = subprocess.check_output(["nvidia-smi", "--id=" + str(gpu),
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    memory, usage = [int(value.strip()) for value in state.strip().split(",")]
    processes = subprocess.check_output(["nvidia-smi", "--id=" + str(gpu),
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if processes or memory >= 1000 or usage > 10:
        raise RuntimeError("GPU {} is busy; refusing to touch other jobs".format(gpu))


def execute(command, gpu, log):
    assert_idle(gpu)
    started = time.time()
    with Path(log).open("x", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, env=environment(gpu), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return time.time() - started


def validate_training(run, epochs):
    run = Path(run)
    meta = read_json(run / "run.json")
    if meta["status"] != "finished" or meta["smoke_test"]:
        raise ValueError("Incomplete or smoke training: " + str(run))
    with (run / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(epochs)):
        raise ValueError("Missing training epochs")
    for row in rows:
        for key in ("eval_top1", "eval_loss", "train_loss"):
            if not math.isfinite(float(row[key])):
                raise ValueError("Non-finite training metric")
    selected = max(rows, key=lambda row: float(row["eval_top1"]))
    return {"checkpoint": str(run / "model_best.pth.tar"),
            "checkpoint_sha256": sha256(run / "model_best.pth.tar"),
            "resolved_sha256": sha256(run / "resolved.yaml"),
            "summary_sha256": sha256(run / "summary.csv"),
            "validation_selected_epoch_zero_based": int(selected["epoch"]),
            "validation_top1": float(selected["eval_top1"])}


def verify_selections(output, protocol):
    output = Path(output)
    records = read_json(output / "selection_frozen_before_test.json")
    expected = set(expected_jobs(protocol))
    if len(records) != len(expected) or {(row["method"], row["seed"]) for row in records} != expected:
        raise ValueError("All 20 selections must be frozen before testing")
    for row in records:
        run = output / "runs" / run_name(row["method"], row["seed"])
        if any(row[key] != value for key, value in validate_training(run, protocol["epochs"]).items()):
            raise ValueError("Selection or completed training changed")
    return records


def host_identity():
    return {"hostname": socket.gethostname(), "user": os.environ.get("USER", "unknown"),
            "ssh_connection": os.environ.get("SSH_CONNECTION", "not recorded"),
            "code_absolute_path": str(ROOT)}
