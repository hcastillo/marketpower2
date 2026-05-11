import unittest

from interbank_web import create_app


class WebMvpTestCase(unittest.TestCase):
    def test_create_app_simulate_index_ok(self):
        app = create_app("simulate")
        client = app.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_simulate_endpoint_returns_expected_keys(self):
        app = create_app("simulate")
        client = app.test_client()
        payload = {"N": 10, "T": 20, "seed": 5, "p": 0.5}
        response = client.post("/api/simulate", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("meta", body)
        self.assertIn("time", body)
        self.assertIn("series", body)
        self.assertEqual(len(body["time"]), 20)

    def test_nan_serialization_uses_null(self):
        app = create_app("simulate")
        client = app.test_client()
        payload = {"N": 10, "T": 30, "seed": 5, "p": 1.0}
        response = client.post("/api/simulate", json=payload)
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertNotIn("NaN", text)

    def test_unimplemented_modes_return_501(self):
        app = create_app("invalid")
        client = app.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 400)


class WebMultipleMvpTestCase(unittest.TestCase):
    def test_create_app_multiple_index_ok(self):
        app = create_app("multiple")
        client = app.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_multiple_endpoint_returns_expected_keys(self):
        app = create_app("multiple")
        client = app.test_client()
        payload = {
            "multiple_param": "p",
            "from": 0,
            "to": 0.2,
            "step": 0.1,
            "mc_seeds": 2,
            "workers": 2,
            "metrics": ["bankruptcies", "equity", "ir"],
            "config": {"N": 10, "T": 30, "seed": 5},
        }
        response = client.post("/api/multiple", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("multiple_param", body)
        self.assertIn("multiple_values", body)
        self.assertIn("metrics", body)
        self.assertEqual(body["multiple_param"], "p")
        self.assertEqual(len(body["multiple_values"]), 3)
        self.assertIn("bankruptcies", body["metrics"])

    def test_multiple_endpoint_rejects_invalid_param(self):
        app = create_app("multiple")
        client = app.test_client()
        payload = {
            "multiple_param": "invalid_param",
            "from": 0,
            "to": 0.2,
            "step": 0.1,
            "metrics": ["bankruptcies"],
            "config": {"N": 10, "T": 30, "seed": 5},
        }
        response = client.post("/api/multiple", json=payload)
        self.assertEqual(response.status_code, 400)


class WebDashboardMvpTestCase(unittest.TestCase):
    def test_create_app_dashboard_index_ok(self):
        app = create_app("dashboard")
        client = app.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_dashboard_supports_simulate_and_multiple(self):
        app = create_app("dashboard")
        client = app.test_client()

        response_sim = client.post("/api/simulate", json={"N": 10, "T": 20, "seed": 5, "p": 0.6})
        self.assertEqual(response_sim.status_code, 200)

        response_multiple = client.post(
            "/api/multiple",
            json={
                "multiple_param": "p",
                "from": 0,
                "to": 0.2,
                "step": 0.1,
                "mc_seeds": 2,
                "workers": 2,
                "metrics": ["bankruptcies", "equity", "ir"],
                "config": {"N": 10, "T": 30, "seed": 5},
            },
        )
        self.assertEqual(response_multiple.status_code, 200)


if __name__ == "__main__":
    unittest.main()
