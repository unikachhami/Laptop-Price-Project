import unittest
from flask_app.app import app


class LaptopPriceAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.client.testing = True

    # ---------------- HOME PAGE TEST ----------------
    def test_home_page(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        # adjust if your HTML has different title/text
        self.assertTrue(
            b"Laptop" in response.data or b"Price" in response.data
        )

    # ---------------- PREDICTION TEST ----------------
    def test_predict_page(self):
        sample_input = {
            "company": "Dell",
            "type": "Notebook",
            "cpu": "Intel Core i5",
            "gpu": "Intel",
            "os": "Windows",

            "ram": 8,
            "weight": 2.2,
            "screen_size": 15.6,

            "resolution": "1920x1080",

            "hdd": 1000,
            "ssd": 256,

            "touchscreen": "No",
            "ips": "Yes"
        }

        response = self.client.post("/predict", data=sample_input)

        self.assertEqual(response.status_code, 200)

        # check output contains something meaningful
        self.assertTrue(
            b"Error" not in response.data,
            "Prediction failed — got error response"
        )

        # optional: check prediction exists in output
        self.assertTrue(
            any(x in response.data.lower() for x in [b"rs", b"price", b"prediction", b""]),
            "Response should contain prediction output"
        )


if __name__ == "__main__":
    unittest.main()