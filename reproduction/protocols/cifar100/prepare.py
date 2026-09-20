"""Check the split and all four training paths, then freeze launch inputs."""
import datetime
import json
import os
from pathlib import Path
import subprocess

from holdout import digest, make_split, load_partition, split_indices
from worker import environment, write_new, PROTOCOL, ROOT, PYTHON, run_name


def main():
    import numpy as np
    labels = np.repeat(np.arange(100), 500)
    first = split_indices(labels)
    assert first == split_indices(labels)
    assert [len(first[name]) for name in ("train", "validation", "internal_test")] == [40000, 5000, 5000]
    make_split(PROTOCOL["data_root"], ROOT)
    generated = json.loads((ROOT / "split.json").read_text())
    reference = json.loads((ROOT / "reference_split.json").read_text())
    if generated['source_sha256'] != reference['source_sha256'] or generated['indices'] != reference['indices']:
        raise ValueError('Dataset or split differs from the archived protocol')
    training, validation = load_partition(ROOT, "train"), load_partition(ROOT, "validation")
    assert len(training) == 40000 and len(validation) == 5000
    training.transform = lambda image: "training_only"
    assert training[0][0] == "training_only" and validation[0][0].size == (32, 32)
    (ROOT / "preflight").mkdir()
    for method in PROTOCOL["methods"]:
        name = "smoke_" + method
        command = [PYTHON, str(ROOT / "holdout.py"), "--config",
                   str(ROOT / "source/configs/cifar100_d8_t4.yaml"),
                   "--method", method, "--data-dir", PROTOCOL["data_root"],
                   "--output", str(ROOT / "preflight"), "--experiment", name, "--smoke"]
        with (ROOT / "preflight" / (name + ".log")).open("x") as stream:
            subprocess.run(command, env=environment(0), cwd=ROOT, check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
        print(name + " passed", flush=True)
    smoke = ROOT / "preflight" / "smoke_B"
    command = [PYTHON, str(ROOT / "evaluate_holdout.py"), "--partition", "validation",
               "--config", str(smoke / "resolved.yaml"), "--checkpoint", str(smoke / "model_best.pth.tar"),
               "--method", "B", "--data-dir", PROTOCOL["data_root"],
               "--output", str(ROOT / "preflight/eval_validation.json"), "--max-batches", "2", "--trusted-checkpoint"]
    with (ROOT / "preflight/eval_validation.log").open("x") as stream:
        subprocess.run(command, env=environment(0), cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    blocked = [PYTHON, str(ROOT / "evaluate_holdout.py"), "--partition", "internal_test"]
    with (ROOT / "preflight/test_gate.log").open("x") as stream:
        result = subprocess.run(blocked, env=environment(0), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode != 0
    assert "selection_frozen_before_test.json" in (ROOT / "preflight/test_gate.log").read_text()
    paths = [ROOT / name for name in ("protocol.json", "split.json", "holdout.py", "evaluate_holdout.py",
                                     "worker.py", "prepare.py", "train.npz", "validation.npz", "internal_test.npz")]
    paths.extend(path for path in (ROOT / "source").rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts)
    write_new(ROOT / "freeze.json", {
        "frozen_at": datetime.datetime.now().astimezone().isoformat(),
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "preflight": "Split balance/disjointness/reproducibility, four method training smokes, validation-only evaluation smoke, and premature-test rejection passed. No internal or official test evaluation."
    })
    records = []
    for config in PROTOCOL["configs"]:
        for method in PROTOCOL["methods"]:
            for seed in PROTOCOL["seeds"]:
                name = run_name(config, method, seed)
                run = ROOT / "runs" / name
                records.append({"server": PROTOCOL["server"], "config": config, "method": method,
                                "seed": seed, "gpu": PROTOCOL["gpu_by_method"][method],
                                "run": str(run), "summary": str(run / "summary.csv"),
                                "checkpoint": str(run / "model_best.pth.tar"),
                                "log": str(ROOT / "logs" / (name + ".log")),
                                "test_result": str(ROOT / "test_results" / (name + ".json"))})
    write_new(ROOT / "RUN_PATHS.json", records)
    with (ROOT / "queue.log").open("x") as stream:
        process = subprocess.Popen([PYTHON, str(ROOT / "worker.py")], cwd=ROOT,
                                   env=environment(0), stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write_new(ROOT / "launch_receipt.json", {"pid": process.pid, "server": PROTOCOL["server"],
              "root": str(ROOT), "launched_at": datetime.datetime.now().astimezone().isoformat()})
    print("Launched detached queue PID " + str(process.pid), flush=True)


if __name__ == "__main__":
    main()
