import unittest
import pandas as pd
from src.preprocessing import handle_missing_values, encode_categorical


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "Hours Studied": [1, 2, None],
            "Previous Scores": [50, None, 70],
            "Extracurricular Activities": ["Yes", "No", "Yes"]
        })

    def test_missing_values(self):
        df_clean = handle_missing_values(self.df)
        self.assertFalse(df_clean.isnull().values.any())

    def test_encoding(self):
        df_encoded = encode_categorical(self.df)
        self.assertTrue(df_encoded["Extracurricular Activities"].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()