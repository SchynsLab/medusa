import torch


class LinearRegression:

    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept

    def fit(self, X, Y):

        if Y.ndim == 3:
            Y = Y.reshape(X.shape[0], -1)

        if self.add_intercept:
            icept = torch.ones((X.shape[0], 1), device=X.device)
            X = torch.hstack([X, icept])

        if X.dtype != torch.float32:
            X = X.float()

        if Y.dtype != torch.float32:
            Y = Y.float()

        self.coef_ = torch.inverse(X.T @ X) @ X.T @ Y

        return self

    def predict(self, X):

        if self.add_intercept:
            icept = torch.ones((X.shape[0], 1), device=X.device)
            X = torch.hstack([X, icept])

        if X.dtype != torch.float32:
            X = X.float()

        Y_hat = (X @ self.coef_).reshape(X.shape[0], -1, 3)

        return Y_hat

    def predict_from_range(self, range_=(-1, 1), steps=100):

        X = torch.linspace(
            range_[0], range_[1], steps=steps, device=self.coef_.device
        ).unsqueeze(1)

        return self.predict(X)
