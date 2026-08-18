import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../../haystack/components/evaluators/production_debt.py",
)
spec = importlib.util.spec_from_file_location("haystack_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["haystack_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtEvaluator = production_debt_mod.ProductionDebtEvaluator
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = ProductionDebtEvaluator(
            never_equate_intent_to_approval=True,
            max_acceptable_pdi=12.0,
        )

    def test_clean_pipeline_passes_readiness(self) -> None:
        report = self.evaluator.run(
            pipeline_id="pipe_enterprise_search_01",
            component_count=5,
            context_tokens=1000,
            generated_tokens=100,
            pipeline_latency_seconds=0.85,
            routing_loop_count=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.pdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_pipeline_fails_debt(self) -> None:
        report = self.evaluator.run(
            pipeline_id="pipe_runaway_routing",
            component_count=12,
            context_tokens=1000,
            generated_tokens=3000,  # High token inflation (4.0x)
            pipeline_latency_seconds=8.5,  # High latency
            routing_loop_count=4,  # 4 routing loops
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.pdi_score, 50.0)
        self.assertIn("HIGH_TOKEN_INFLATION_4.00X", report.critical_smells)
        self.assertIn("HIGH_PIPELINE_LATENCY_8.50S", report.critical_smells)
        self.assertIn("DETECTED_4_ROUTING_LOOPS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.evaluator.run("pipe-1")
        self.evaluator.run("pipe-2")
        self.evaluator.run("pipe-3")

        entries = self.evaluator.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.evaluator.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
