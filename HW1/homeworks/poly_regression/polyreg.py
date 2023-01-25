"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields


    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        return X ** np.arange(1, degree + 1)

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        # get the model 
        X_ = self.polyfeatures(X, self.degree)
        # standardize
        self.mean = X_.mean(axis=0)
        self.std = X_.std(axis=0)
        X_ = (X_ - self.mean) / self.std
        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X_]
        n, d = X_.shape
        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d)
        reg_matrix[0, 0] = 0
        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.solve(X_.T @ X_ + reg_matrix , X_.T @ y )

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_ = np.c_[np.ones([n, 1]), (self.polyfeatures(X, self.degree) - self.mean) / self.std]
        return X_ @ self.weight

@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    return ((a - b) ** 2).sum() / b.shape[0]

@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    # Fill in errorTrain and errorTest arrays
    for i in range(len(Xtrain)):
        error = PolynomialRegression(degree, reg_lambda) # initial error model 
        error.fit(Xtrain[0:i+1], Ytrain[0:i+1]) # fit error model 
        errorTrain[i] = mean_squared_error(error.predict(Xtrain[0:i+1]), Ytrain[0:i+1])
        errorTest[i] = mean_squared_error(error.predict(Xtest), Ytest)
    return errorTrain, errorTest

