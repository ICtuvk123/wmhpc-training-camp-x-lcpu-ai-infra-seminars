"""Host-only checks: malformed/failed measurements must never become PASS."""

import unittest

from benchmark_p1 import OCCUPANCY_METRIC, build_metrics, parse_occupancy
from check_p1 import cases, parse_record, validate


class P1HarnessTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(stage="p1", implementation="tcgen05", cc="10.3", correct=True,
                           state_unchanged=True, batch=3, checked_ctas=3,
                           input="zero", state="tagged", exact=True, seed=2026,
                           max_abs=0.0, max_rel=0.0, launch_us=1.0, cta_ns=333.0)
        self.flags = ["--input", "zero", "--state", "tagged"]

    def test_valid_state_only_case(self):
        validate(self.record, "tcgen05", 3, self.flags)

    def test_rejects_false_or_incomplete_pass(self):
        for change in [dict(correct=False), dict(state_unchanged=False), dict(checked_ctas=1),
                       dict(stage="p0"), dict(cc="8.9"), dict(state="zero"), dict(seed=1),
                       dict(max_abs=1e-9), dict(max_abs=float("nan")),
                       dict(launch_us=float("inf")), dict(batch=1), dict(exact=False)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(dict(self.record, **change), "tcgen05", 3, self.flags)

    def test_off_diagonal_coordinates_checked(self):
        record = dict(self.record, input="one-hot", probe_m=5, probe_k=3, probe_n=37)
        flags = ["--input", "one-hot", "--state", "tagged", "--m", "5", "--k", "3", "--n", "37"]
        validate(record, "tcgen05", 3, flags)
        record["probe_n"] = 5
        with self.assertRaises(ValueError):
            validate(record, "tcgen05", 3, flags)

    def test_ambiguous_result_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_record('noise\n{"implementation":"baseline"}\n{"implementation":"tcgen05"}')

    def test_regressions_cover_state_and_reduction_axes(self):
        flags = [dict(zip(values[::2], values[1::2])) for _, values in cases()]
        self.assertEqual({int(f["--k"]) for f in flags if "--k" in f}, set(range(16)))
        self.assertEqual({f["--state"] for f in flags if f["--input"] == "zero"},
                         {"zero", "tagged", "random"})

    def test_spills_report_unknown_separately_from_zero(self):
        self.assertIsNone(build_metrics("")["spill_load_bytes"])
        metrics = build_metrics("Used 154 registers\n0 bytes stack frame, "
                                "16 bytes spill stores, 8 bytes spill loads")
        self.assertEqual(metrics["ptxas_registers"], 154)
        self.assertEqual(metrics["spill_store_bytes"], 16)
        self.assertEqual(metrics["spill_load_bytes"], 8)

    def test_occupancy_csv_not_confused_with_application_json(self):
        output = ('==PROF== Connected\n{"implementation":"tcgen05"}\n'
                  '"ID","Metric Name","Metric Unit","Metric Value"\n'
                  f'"0","{OCCUPANCY_METRIC}","%","14.73"\n')
        self.assertEqual(parse_occupancy(output), 14.73)
        with self.assertRaises(ValueError):
            parse_occupancy(output.replace("14.73", "nan"))
        with self.assertRaises(ValueError):
            parse_occupancy(output + f'"1","{OCCUPANCY_METRIC}","%","18.00"\n')
        with self.assertRaises(ValueError):
            parse_occupancy("==ERROR== ERR_NVGPUCTRPERM")


if __name__ == "__main__":
    unittest.main()
