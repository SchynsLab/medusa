import torch


class LinearRegression:
    """Linear regression model with normal equations, using PyTorch.

    Parameters
    ----------
    add_intercept : bool
        Whether to add an intercept term to the model

    Attributes
    ----------
    coef_ : torch.tensor
        The model coefficients, including the intercept if applicable
    """
    def __init__(self, add_intercept=True):
        """Initializes the model."""
        self.add_intercept = add_intercept

    def fit(self, X, Y):
        """Fits the model to the data (X, Y).

        Parameters
        ----------
        X : torch.tensor
            The input data, with shape (n_samples, n_features)
        Y : torch.tensor
            The output data, with shape (n_samples, n_outputs)
        """
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
        """Predicts the output for the given input data.

        Parameters
        ----------
        X : torch.tensor
            The input data, with shape (n_samples, n_features)

        Returns
        -------
        Y_hat : torch.tensor
            The predicted output data, with shape (n_samples, n_outputs)
        """
        if self.add_intercept:
            icept = torch.ones((X.shape[0], 1), device=X.device)
            X = torch.hstack([X, icept])

        if X.dtype != torch.float32:
            X = X.float()

        Y_hat = (X @ self.coef_).reshape(X.shape[0], -1, 3)

        return Y_hat

    def predict_from_range(self, range_=(-1, 1), steps=100):
        """Predicts the output for the given input range.

        Parameters
        ----------
        range_ : tuple
            The range of input values to predict over
        steps : int
            The number of steps to take between the range's start and end

        Returns
        -------
        Y_hat : torch.tensor
            The predicted output data, with shape (n_samples, n_outputs)
        """
        X = torch.linspace(
            range_[0], range_[1], steps=steps, device=self.coef_.device
        ).unsqueeze(1)
        Y_hat = self.predict(X)

        return Y_hat
