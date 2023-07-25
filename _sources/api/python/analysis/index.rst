:py:mod:`medusa.analysis`
=========================

.. py:module:: medusa.analysis


Module Contents
---------------

.. py:class:: LinearRegression(add_intercept=True)

   Linear regression model with normal equations, using PyTorch.

   :param add_intercept: Whether to add an intercept term to the model
   :type add_intercept: bool

   .. attribute:: coef_

      The model coefficients, including the intercept if applicable

      :type: torch.tensor

   .. py:method:: fit(X, Y)

      Fits the model to the data (X, Y).

      :param X: The input data, with shape (n_samples, n_features)
      :type X: torch.tensor
      :param Y: The output data, with shape (n_samples, n_outputs)
      :type Y: torch.tensor


   .. py:method:: predict(X)

      Predicts the output for the given input data.

      :param X: The input data, with shape (n_samples, n_features)
      :type X: torch.tensor

      :returns: **Y_hat** -- The predicted output data, with shape (n_samples, n_outputs)
      :rtype: torch.tensor


   .. py:method:: predict_from_range(range_=(-1, 1), steps=100)

      Predicts the output for the given input range.

      :param range_: The range of input values to predict over
      :type range_: tuple
      :param steps: The number of steps to take between the range's start and end
      :type steps: int

      :returns: **Y_hat** -- The predicted output data, with shape (n_samples, n_outputs)
      :rtype: torch.tensor



