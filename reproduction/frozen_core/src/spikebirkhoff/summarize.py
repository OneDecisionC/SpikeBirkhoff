"""Summarize monitored test logs and verify the published result manifest.

Copyright (c) 2026 Zhiqi Cai. Licensed under the MIT License.
This module uses only the Python standard library.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path


EXPECTED_EPOCHS = 210
METRICS = ("peak_test", "late20", "final")
COLUMNS = (
    "experiment_id", "configuration_id", "method", "seed", "epochs",
    "status", "peak_test", "late20", "final", "first_peak_epoch",
    "last_observed",
)


class ResultError(ValueError):
    """A log or its published result record failed validation."""


def _accuracy(value, label):
    if not isinstance(value, str):
        raise ResultError("{} must be a decimal string".format(label))
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ResultError("{} is not a decimal: {!r}".format(label, value)) from exc
    if not number.is_finite() or not Decimal(0) <= number <= Decimal(100):
        raise ResultError("{} must be finite and between 0 and 100".format(label))
    return number


def _decimal_text(value):
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def summarize_csv(path):
    """Read one UTF-8 log; require ordered, contiguous epochs beginning at zero.

    A complete paper run contains exactly 210 epochs. For another run length,
    peak_test covers the observed rows, while late20 and final remain unset.
    The SHA-256 and metrics are calculated from the same snapshot of the file.
    """
    path = Path(path)
    data = path.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    except UnicodeError as exc:
        raise ResultError("{}: CSV must be UTF-8".format(path)) from exc
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ResultError("{}: duplicate CSV column names".format(path))
    if not {"epoch", "eval_top1"}.issubset(fields):
        raise ResultError("{}: CSV requires epoch and eval_top1 columns".format(path))

    values = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ResultError("{}: malformed CSV row {}".format(path, reader.line_num))
        raw_epoch = row["epoch"].strip()
        if not re.fullmatch(r"[0-9]+", raw_epoch):
            raise ResultError("{}: invalid epoch {!r}".format(path, raw_epoch))
        epoch = int(raw_epoch)
        if epoch != len(values):
            raise ResultError(
                "{}: expected epoch {}, found {}; epochs must be ordered and contiguous"
                .format(path, len(values), epoch)
            )
        values.append(_accuracy(row["eval_top1"], "{} epoch {} eval_top1".format(path, epoch)))
    if not values:
        raise ResultError("{}: CSV has no epoch rows".format(path))

    complete = len(values) == EXPECTED_EPOCHS
    peak = max(values)
    late20 = None
    if complete:
        tail = values[190:210]
        # Preserve all input decimal digits when adding and dividing by 20.
        with localcontext() as context:
            context.prec = max(
                50,
                max(len(v.as_tuple().digits) + abs(v.as_tuple().exponent) for v in tail) + 10,
            )
            late20 = _decimal_text(sum(tail, Decimal(0)) / Decimal(20))
    return {
        "experiment_id": path.stem,
        "configuration_id": "",
        "method": "",
        "seed": "",
        "epochs": len(values),
        "status": "COMPLETE" if complete else "INCOMPLETE ({}/210 epochs)".format(len(values)),
        "peak_test": _decimal_text(peak),
        "late20": late20,
        "final": _decimal_text(values[209]) if complete else None,
        "first_peak_epoch": values.index(peak),
        "last_observed": _decimal_text(values[-1]),
        "csv": str(path),
        "csv_sha256": hashlib.sha256(data).hexdigest(),
    }


def _validate_record(record):
    if not isinstance(record, dict):
        raise ResultError("each result must be an object")
    for key in ("experiment_id", "configuration_id", "method", "csv", "csv_sha256"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ResultError("{} must be a nonempty string".format(key))
    for key in ("seed", "epochs", "first_peak_epoch"):
        if type(record.get(key)) is not int:
            raise ResultError("{} must be an integer".format(key))
    if record["epochs"] != EXPECTED_EPOCHS:
        raise ResultError("published results require exactly 210 epochs")
    if not 0 <= record["first_peak_epoch"] < EXPECTED_EPOCHS:
        raise ResultError("first_peak_epoch must be between 0 and 209")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", record["csv_sha256"]):
        raise ResultError("csv_sha256 must contain 64 hexadecimal characters")
    for key in METRICS:
        _accuracy(record.get(key), key)


def load_results(path, verify=False):
    """Load schema-v1 results, optionally auditing every source CSV.

    Manifest CSV paths are relative to the repository root: the parent of the
    directory containing results/paper_results.json. No current-directory
    fallback is used, so verification also works outside the repository.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ResultError("unsupported result schema; expected schema_version 1")
    records = manifest.get("results")
    if not isinstance(records, list) or not records:
        raise ResultError("results must be a nonempty list")

    summaries, errors, identifiers = [], [], set()
    for index, record in enumerate(records):
        label = record.get("experiment_id", "result {}".format(index)) if isinstance(record, dict) else "result {}".format(index)
        try:
            _validate_record(record)
            if record["experiment_id"] in identifiers:
                raise ResultError("duplicate experiment_id")
            identifiers.add(record["experiment_id"])
            if verify:
                actual = summarize_csv(path.parent.parent / record["csv"])
                mismatches = []
                if actual["csv_sha256"] != record["csv_sha256"].lower():
                    mismatches.append("CSV SHA-256 mismatch")
                for key in ("epochs", "first_peak_epoch"):
                    if actual[key] != record[key]:
                        mismatches.append("{}: recorded {}, observed {}".format(key, record[key], actual[key]))
                for key in METRICS:
                    if actual[key] is None or Decimal(actual[key]) != Decimal(record[key]):
                        mismatches.append("{}: recorded {}, observed {}".format(key, record[key], actual[key]))
                if mismatches:
                    raise ResultError("; ".join(mismatches))
                summary = dict(record, status="VERIFIED", last_observed=actual["last_observed"])
            else:
                summary = dict(record, status="UNVERIFIED", last_observed=None)
            summaries.append(summary)
        except (OSError, ValueError, csv.Error) as exc:
            errors.append("{}: {}".format(label, exc))
    if errors:
        raise ResultError("Result verification failed:\n" + "\n".join(errors))
    return summaries


def render_summary(rows, output_format="markdown"):
    """Return an exact-decimal CSV or a Markdown summary."""
    if output_format == "csv":
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()
    if output_format != "markdown":
        raise ResultError("output format must be csv or markdown")

    def cell(value):
        if value is None or value == "":
            return "—"
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    lines = [
        "Monitored test accuracy (%). Peak values are selected from training-time test monitoring.",
        "For complete runs: late20 = mean of epochs 190–209; final = epoch 209.",
        "Incomplete logs show an observed peak and last_observed; paper late20/final are unset.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    lines.extend("| " + " | ".join(cell(row.get(key)) for key in COLUMNS) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, help="your own schema-v1 results manifest")
    source.add_argument("--csv", type=Path, help="summarize one new epoch/eval_top1 CSV log")
    parser.add_argument("--verify", action="store_true", help="verify all published CSV hashes, epochs, and decimal metrics")
    parser.add_argument("--output", type=Path, help="also write the summary to a .csv, .md, or .markdown file")
    args = parser.parse_args(argv)
    if args.csv and args.verify:
        parser.error("--verify requires --results; a single --csv has no reference manifest")
    if args.output and args.output.suffix.lower() not in (".csv", ".md", ".markdown"):
        parser.error("--output must end in .csv, .md, or .markdown")

    try:
        rows = [summarize_csv(args.csv)] if args.csv else load_results(
            args.results, verify=args.verify
        )
        summary = render_summary(rows)
        if args.output:
            output_format = "csv" if args.output.suffix.lower() == ".csv" else "markdown"
            args.output.write_text(render_summary(rows, output_format), encoding="utf-8")
        print(summary, end="")
        if args.verify:
            print("Verified {} logs: SHA-256, epochs 0–209, and exact decimal metrics.".format(len(rows)))
        return 0
    except (OSError, ValueError, csv.Error) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
