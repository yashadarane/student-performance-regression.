from sklearn.linear_model import LinearRegression

class PerformancePredictor:
    def __init__(self):
        """Initializes the Linear Regression model."""
        self.model = LinearRegression()

    def train(self, X, y):
        """Trains the model using provided features and target."""
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Makes predictions on new data."""
        return self.model.predict(X)