"""Frozen four-GPU experiment queue with an all-runs-complete test gate."""
import concurrent.futures
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from holdout import digest

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
PYTHON = PROTOCOL["python"]


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def environment(gpu):
    result = dict(os.environ)
    result.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / "source" / "src"),
                  OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  PYTHONUNBUFFERED="1", PYTHONHASHSEED="0")
    return result


def run_name(config, method, seed):
    return "{}_{}_seed{}".format(config, method, seed)


def train_method(method):
    gpu = PROTOCOL["gpu_by_method"][method]
    completed = []
    for config in PROTOCOL["configs"]:
        for seed in PROTOCOL["seeds"]:
            name = run_name(config, method, seed)
            log = ROOT / "logs" / (name + ".log")
            command = [PYTHON, str(ROOT / "holdout.py"), "--config",
                       str(ROOT / "source" / "configs" / (config + ".yaml")),
                       "--method", method, "--data-dir", PROTOCOL["data_root"],
                       "--output", str(ROOT / "runs"), "--experiment", name,
                       "--set", "seed={}".format(seed)]
            for key, value in PROTOCOL["operational_overrides"].items():
                command += ["--set", "{}={}".format(key, value)]
            started = time.time()
            with log.open("x") as stream:
                process = subprocess.Popen(command, env=environment(gpu), cwd=ROOT,
                                           stdout=stream, stderr=subprocess.STDOUT)
                write_new(ROOT / "receipts" / (name + ".started.json"),
                          {"pid": process.pid, "gpu": gpu, "command": command,
                           "started_unix": started, "log": str(log),
                           "server": PROTOCOL["server"], "run": str(ROOT / "runs" / name)})
                returncode = process.wait()
            write_new(ROOT / "receipts" / (name + ".finished.json"),
                      {"returncode": returncode, "elapsed_seconds": time.time() - started})
            if returncode:
                raise RuntimeError("Training failed; no test evaluation: " + name)
            with (ROOT / "runs" / name / "summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            if [int(row["epoch"]) for row in rows] != list(range(PROTOCOL["epochs"])):
                raise RuntimeError("Incomplete training: " + name)
            selected = max(rows, key=lambda row: float(row["eval_top1"]))
            checkpoint = ROOT / "runs" / name / "model_best.pth.tar"
            completed.append({"name": name, "config": config, "method": method, "seed": seed,
                              "gpu": gpu, "checkpoint": str(checkpoint),
                              "checkpoint_sha256": digest(checkpoint),
                              "validation_selected_epoch_zero_based": int(selected["epoch"]),
                              "validation_top1": float(selected["eval_top1"])})
    return completed


def evaluate(record):
    name = record["name"]
    if digest(record["checkpoint"]) != record["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint changed")
    run = ROOT / "runs" / name
    output = ROOT / "test_results" / (name + ".json")
    command = [PYTHON, str(ROOT / "evaluate_holdout.py"), "--partition", "internal_test", "--config", str(run / "resolved.yaml"),
               "--checkpoint", record["checkpoint"], "--method", record["method"],
               "--data-dir", PROTOCOL["data_root"], "--output", str(output),
               "--workers", "4", "--trusted-checkpoint"]
    with (ROOT / "logs" / (name + ".test.log")).open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=environment(record["gpu"]), check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
    result = json.loads(output.read_text())
    if not result["complete_test_set"] or result["samples"] != 5000 or result["evaluation_partition"] != "internal_test":
        raise RuntimeError("Incomplete internal test evaluation")
    return dict(record, test_top1=result["top1"], result_path=str(output))


def main():
    lock = (ROOT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze = json.loads((ROOT / "freeze.json").read_text())
    for relative, expected in freeze["sha256"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError("Frozen input changed: " + relative)
    gpu_rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip().splitlines()
    for row in gpu_rows:
        gpu, memory, usage = [int(value.strip()) for value in row.split(",")]
        if gpu in PROTOCOL["gpu_by_method"].values() and (memory > 1000 or usage > 10):
            raise RuntimeError("GPU is not idle; refusing to interfere: " + str(gpu))
    for folder in ("logs", "receipts", "runs", "test_results"):
        (ROOT / folder).mkdir(exist_ok=True)
    write_new(ROOT / "queue_started.json", {"pid": os.getpid(), "time": datetime.datetime.now().astimezone().isoformat(),
                                          "protocol_sha256": digest(ROOT / "protocol.json")})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(train_method, PROTOCOL["methods"]))
    selected = [record for group in groups for record in group]
    if len(selected) != len(PROTOCOL['configs']) * len(PROTOCOL['methods']) * len(PROTOCOL['seeds']):
        raise RuntimeError("Refusing premature test evaluation")
    write_new(ROOT / "selection_frozen_before_test.json", selected)
    results = [evaluate(record) for record in selected]
    summary = {"evaluation_partition": "internal_test_5000_from_original_training",
               "official_benchmark": False, "per_run": results, "aggregates": [], "paired_differences": []}
    for config in PROTOCOL["configs"]:
        for method in PROTOCOL["methods"]:
            values = [record["test_top1"] for record in results
                      if record["config"] == config and record["method"] == method]
            summary["aggregates"].append({"config": config, "method": method, "n": len(values),
                                          "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)})
        indexed = {(record["method"], record["seed"]): record["test_top1"]
                   for record in results if record["config"] == config}
        for control in ("A0", "S", "I"):
            differences = [indexed["B", seed] - indexed[control, seed] for seed in PROTOCOL["seeds"]]
            summary["paired_differences"].append({"config": config, "contrast": "B-" + control,
                                                  "values_pp": differences, "mean_pp": statistics.mean(differences),
                                                  "sample_sd_pp": statistics.stdev(differences)})
    write_new(ROOT / "RESULTS.json", summary)
    print("All frozen runs and test evaluations completed.", flush=True)


if __name__ == "__main__":
    main()
