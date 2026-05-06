import unittest
from flask_app.app import app


class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    # -----------------------------
    # HOME PAGE TEST
    # -----------------------------
    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    # -----------------------------
    # PREDICT ROUTE TEST
    # -----------------------------
    def test_predict_route(self):

        payload = {
            "company": "Apple",
            "type": "Ultrabook",
            "cpu": "Intel Core i5",
            "gpu": "Intel",
            "os": "Mac",
            "ram": "8",
            "weight": "1.5",
            "screen_size": "15.6",
            "resolution": "1920x1080",
            "hdd": "0",
            "ssd": "256",
            "touchscreen": "No",
            "ips": "Yes"
        }

        response = self.client.post("/predict", data=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"result", response.data)


if __name__ == "__main__":
    unittest.main()