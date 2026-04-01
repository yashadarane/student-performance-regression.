import unittest
import pandas as pd
from src.model import train_model, predict


class TestModel(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
            "Hours Studied": [1, 2, 3],
            "Previous Scores": [50, 60, 70],
            "Sleep Hours": [6, 7, 8],
            "Practice Papers Solved": [1, 2, 3],
            "Extracurricular Activities": [1, 0, 1]
        })

        self.y = pd.Series([55, 65, 75])

    def test_training(self):
        model = train_model(self.X, self.y)
        self.assertIsNotNone(model)

    def test_prediction(self):
        model = train_model(self.X, self.y)
        preds = predict(model, self.X)
        self.assertEqual(len(preds), len(self.X))


if __name__ == "__main__":
    unittest.main()