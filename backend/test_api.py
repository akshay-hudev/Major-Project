"""Unit and integration tests for the Cardiac Progression FastAPI backend."""

import unittest
from fastapi.testclient import TestClient
from backend.main import app
from backend.patient_presets import ALL_SCENARIOS


class TestCardiacProgressionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_status_endpoint(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "operational")
        self.assertEqual(data["feature_count"], 59)
        self.assertEqual(data["sequence_length"], 6)
        self.assertIn("MIMIC-IV", data["dataset"])

    def test_features_dictionary(self):
        resp = self.client.get("/api/features")
        self.assertEqual(resp.status_code, 200)
        features = resp.json()
        self.assertEqual(len(features), 59)
        feat_ids = {f["id"] for f in features}
        self.assertIn("heart_rate_mean", feat_ids)
        self.assertIn("troponin_t_mean", feat_ids)
        self.assertIn("bnp_mean", feat_ids)
        self.assertIn("creatinine_mean", feat_ids)
        self.assertIn("spo2_mean", feat_ids)

    def test_scenarios_endpoint(self):
        resp = self.client.get("/api/scenarios")
        self.assertEqual(resp.status_code, 200)
        scenarios = resp.json()
        self.assertGreaterEqual(len(scenarios), 5)
        first = scenarios[0]
        self.assertIn("id", first)
        self.assertIn("sample_data", first)
        self.assertEqual(len(first["sample_data"]["visits"]), 6)

    def test_scenario_by_id(self):
        resp = self.client.get("/api/scenarios/DEMO-001")
        self.assertEqual(resp.status_code, 200)
        scenario = resp.json()
        self.assertEqual(scenario["id"], "DEMO-001")

        # Test non-existent scenario
        resp404 = self.client.get("/api/scenarios/DOES-NOT-EXIST")
        self.assertEqual(resp404.status_code, 404)

    def test_predict_stable_patient(self):
        stable_scenario = ALL_SCENARIOS[0]
        payload = stable_scenario["sample_data"]

        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertIn("probability", res)
        self.assertIn("prediction", res)
        self.assertLess(res["probability"], 0.335)
        self.assertEqual(res["prediction"], 0)
        self.assertEqual(res["risk_level"], "Low")
        self.assertGreaterEqual(len(res["top_contributors"]), 1)
        self.assertGreaterEqual(len(res["recommendations"]), 1)

    def test_predict_worsening_patient(self):
        decomp_scenario = ALL_SCENARIOS[1]  # Acute Decompensated Heart Failure
        payload = decomp_scenario["sample_data"]

        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertGreater(res["probability"], 0.335)
        self.assertEqual(res["prediction"], 1)
        self.assertIn(res["risk_level"], ["High", "Critical"])
        self.assertGreaterEqual(len(res["top_contributors"]), 1)
        # Check that leading signals flag deterioration
        driver_names = [c["feature_name"] for c in res["top_contributors"]]
        self.assertTrue(any("BNP" in name or "SpO" in name or "Systolic" in name for name in driver_names))

    def test_predict_invalid_visit_count(self):
        # Provide only 4 visits instead of 6
        payload = {
            "patient_id": "PT-INVALID",
            "visits": ALL_SCENARIOS[0]["sample_data"]["visits"][:4],
        }
        resp = self.client.post("/api/predict", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Exactly 6 visits are required", resp.json()["detail"])

    def test_batch_predict(self):
        batch_payload = {
            "patients": [
                {
                    "patient_id": "PT-BATCH-1",
                    "visits": ALL_SCENARIOS[0]["sample_data"]["visits"],
                },
                {
                    "patient_id": "PT-BATCH-2",
                    "visits": ALL_SCENARIOS[1]["sample_data"]["visits"],
                },
            ]
        }
        resp = self.client.post("/api/predict/batch", json=batch_payload)
        self.assertEqual(resp.status_code, 200)
        batch_res = resp.json()
        self.assertEqual(batch_res["total_processed"], 2)
        self.assertEqual(batch_res["worsening_count"], 1)
        self.assertEqual(batch_res["stable_count"], 1)

    def test_benchmarks_endpoint(self):
        resp = self.client.get("/api/benchmarks")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        self.assertIn("figures", data)
        self.assertGreaterEqual(len(data["models"]), 8)

    def test_sample_csv_endpoint(self):
        resp = self.client.get("/api/sample-csv")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "text/csv; charset=utf-8")
        content = resp.text
        self.assertIn("patient_id", content)
        self.assertIn("heart_rate_mean", content)


if __name__ == "__main__":
    unittest.main()
