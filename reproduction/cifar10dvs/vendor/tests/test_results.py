"""Check log summarization using synthetic fixtures without ML dependencies."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from spikebirkhoff.summarize import ResultError, load_results, main, summarize_csv


REPOSITORY = Path(__file__).resolve().parents[1]


class ResultAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "results").mkdir()
        self.log = self.root / "results" / "run.csv"
        self.manifest = self.root / "results" / "paper_results.json"

    def write_log(self, epochs=range(210), values=None):
        with self.log.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("epoch", "eval_top1"))
            for epoch in epochs:
                writer.writerow((epoch, values[epoch] if values is not None else "0.3" if epoch in (10, 190, 209) else "0.1"))

    def write_manifest(self, **changes):
        record = {
            "experiment_id": "decimal_test", "configuration_id": "test",
            "method": "standard", "seed": 42, "epochs": 210,
            "peak_test": "0.3", "late20": "0.12", "final": "0.3",
            "first_peak_epoch": 10, "csv": "results/run.csv",
            "csv_sha256": hashlib.sha256(self.log.read_bytes()).hexdigest(),
        }
        record.update(changes)
        self.manifest.write_text(json.dumps({"schema_version": 1, "results": [record]}), encoding="utf-8")

    def test_synthetic_manifest_matches_exact_audit(self):
        self.write_log()
        self.write_manifest()
        rows = load_results(self.manifest, verify=True)
        self.assertEqual(len(rows), 1)
        self.assertTrue(all(row["status"] == "VERIFIED" and row["epochs"] == 210 for row in rows))

    def test_decimal_average_and_first_peak_tie(self):
        self.write_log()
        self.write_manifest()
        actual = summarize_csv(self.log)
        self.assertEqual(actual["late20"], "0.12")
        self.assertEqual(actual["first_peak_epoch"], 10)
        self.assertEqual(load_results(self.manifest, verify=True)[0]["status"], "VERIFIED")

    def test_sha_detects_change_that_does_not_change_metrics(self):
        self.write_log()
        self.write_manifest()
        with self.log.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ResultError, "SHA-256 mismatch"):
            load_results(self.manifest, verify=True)

    def test_every_recorded_metric_is_checked_without_rounding(self):
        self.write_log()
        for key, value in (("peak_test", "0.300000000000000001"), ("late20", "0.120000000000000001"), ("final", "0.2"), ("first_peak_epoch", 190)):
            with self.subTest(metric=key):
                self.write_manifest(**{key: value})
                with self.assertRaisesRegex(ResultError, key):
                    load_results(self.manifest, verify=True)

    def test_missing_duplicate_and_reordered_epochs_are_rejected(self):
        valid = list(range(210))
        reordered = valid.copy()
        reordered[40], reordered[41] = reordered[41], reordered[40]
        for epochs in (valid[:40] + valid[41:], valid[:40] + [39] + valid[41:], reordered):
            with self.subTest(epochs=epochs[39:42]):
                self.write_log(epochs)
                self.write_manifest()  # A fresh valid hash must not conceal invalid epochs.
                with self.assertRaisesRegex(ResultError, "ordered and contiguous"):
                    load_results(self.manifest, verify=True)

    def test_incomplete_and_overlong_logs_do_not_claim_paper_metrics(self):
        for length in (20, 209, 211):
            with self.subTest(length=length):
                self.write_log(range(length))
                actual = summarize_csv(self.log)
                self.assertIn("INCOMPLETE", actual["status"])
                self.assertIsNone(actual["late20"])
                self.assertIsNone(actual["final"])
                self.assertIsNotNone(actual["last_observed"])
                self.write_manifest()
                with self.assertRaisesRegex(ResultError, "epochs:"):
                    load_results(self.manifest, verify=True)

    def test_invalid_and_nonfinite_accuracy_are_rejected(self):
        for value in ("NaN", "Infinity", "-0.1", "100.1", "not-a-number", ""):
            with self.subTest(value=value):
                self.write_log(range(1), [value])
                with self.assertRaises(ResultError):
                    summarize_csv(self.log)

    def test_cli_output_and_failure_exit_status(self):
        self.write_log()
        self.write_manifest()
        output = self.root / "summary.csv"
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--results", str(self.manifest), "--verify", "--output", str(output)])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("Verified 1 logs", stdout.getvalue())
        with output.open(encoding="utf-8", newline="") as stream:
            self.assertEqual(next(csv.DictReader(stream))["late20"], "0.12")
        self.write_manifest(final="0.2")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["--results", str(self.manifest), "--verify"]), 1)
        self.assertIn("final:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
