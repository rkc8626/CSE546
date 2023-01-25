from os import times
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    return 2 * np.sum(X ** 2, axis = 0)

@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    return  ((np.linalg.norm(X.dot(weight) + bias - y))**2 
            + _lambda * (np.linalg.norm(weight, ord=1))) 

@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    bias = np.average(y - X @ weight)
    for k in range(X.shape[1]):
        ck = 2 * X[:, k] @ (y - (bias + X @ weight - X[:, k] * weight[k]))
        if ck < -_lambda:  
            weight[k] = (ck + _lambda) / a[k]
        elif ck > _lambda: 
            weight[k] = (ck - _lambda) / a[k]
        else: weight[k] = 0.0
    return weight, bias

@problem.tag("hw2-A", start_line=4)
def train( X: np.ndarray, y: np.ndarray, _lambda: float = 0.01, 
convergence_delta: float = 1e-4, start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None
    # initializing
    while not convergence_criterion(start_weight, old_w, convergence_delta):  
        old_w = np.copy(start_weight) # renew the array
        weight, bias = step(X, y, start_weight, a, _lambda)

    return weight, bias

@problem.tag("hw2-A")
def convergence_criterion(weight: np.ndarray, old_w: np.ndarray, 
convergence_delta: float) -> bool:
    if old_w is None: return False
    else: return ((abs(max(old_w) - max(weight))) < convergence_delta)



@problem.tag("hw2-A")
def main():
    np.random.seed(10000) # use random seed to get array
    # setting based on question
    n, d, k, sigma = 500, 1000, 100, 1
    points = 50
    weight = np.zeros((d, ))
    for j in range(1, k+1):
        weight[j-1] = j/k
    X = np.random.normal(size=(n, d))
    y = X.dot(weight) + np.random.normal(size=(n,))
    reg_lambda = max(2*np.sum(X * (y-np.mean(y)) [:, None], axis=0))
    # initilazing arrays for plotting
    lambdas, nonzeros, fdrs, tprs = [], [], [], [] 
    # compute those arrays
    for _ in range(points):
        f_weight, f_bias = train(X, y, reg_lambda, 1e-3, weight)
        lambdas.append(reg_lambda)

        nonzero = np.sum(abs(f_weight) > 0)
        fdr = np.sum(abs(f_weight[k:]) > 0) / nonzero
        tpr = np.sum(abs(f_weight[:k]) > 0) / k

        nonzeros.append(nonzero)
        fdrs.append(fdr)
        tprs.append(tpr)
        reg_lambda /= 2
    
    # plot A5a
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, nonzeros)
    plt.xscale('log')
    plt.title("Number of Non-zeros vs $\lambda$")
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-zeros')
    plt.savefig('A5a.png')
    # plot A5b
    plt.figure(figsize=(7, 5))
    plt.plot(fdrs, tprs)
    plt.title("False Discovery Rate vs True Positive Rate")
    plt.xlabel('False Discovery Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('A5b.png')

if __name__ == '__main__':
    main()