if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_dataset, problem

@problem.tag("hw2-A", start_line=3)
def main():
    df_train, df_test = load_dataset("crime")
    df_train, df_test= pd.read_table("crime-train.txt"), pd.read_table("crime-test.txt")
    X, y= df_train.drop('ViolentCrimesPerPop', axis=1), df_train['ViolentCrimesPerPop']
    X_test, y_test = df_test.drop('ViolentCrimesPerPop', axis=1), df_test['ViolentCrimesPerPop']
    reg_lambda = max(2 * np.sum(X.T * (y - np.mean(y)), axis=0))
    print(reg_lambda)
    
    # use to test Lasso coefficien, while set lamda = 30 
    f_weight, f_bias = train(X_test.values, y_test.values, 30)    
    print('Largest feacture: ', X.columns[np.argmax(f_weight)], max(f_weight))
    print('Smallest feacture: ', X.columns[np.argmin(f_weight)], min(f_weight))
    
    points = 50 
    # initilazing arrays for plotting
    lambdas, nonzeros, mse_train, mse_test, reg_path = [], [], [], [], []
    for i in range(points):
        lambdas.append(reg_lambda)
        f_weight_train, f_bias_train = train(X.values, y.values, reg_lambda, 1e-3)
        f_weight_test, f_bias_test = train(X_test.values, y_test.values, reg_lambda, 1e-3)

        y_train_pred = X.values.dot(f_weight_train) + f_bias_train
        y_test_pred = X_test.values.dot(f_weight_test) + f_bias_test
        
        nonzeros.append(np.sum(abs(f_weight_train) > 0))      
        mse_train.append(np.mean((y.values-y_train_pred)**2))
        mse_test.append(np.mean((y_test.values-y_test_pred)**2))
        
        reg_lambda /= 2
    # plot for Number of Non-zeros
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, nonzeros)
    plt.xscale('log')
    plt.title("$\lambda$ vs Non Zero Features")
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-zeros')
    plt.savefig('A6c.png')
    # plot for $\lambda$ vs Error
    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, mse_train, label='Train')
    plt.plot(lambdas, mse_test, label='Test')
    plt.title("$\lambda$ vs Error")
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig('A6e.png')
    # plot for Regularization path, Feature weights vs $\lambda$
    plt.figure(figsize=(7, 5))
    vis_feat_name = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    for name in vis_feat_name:
        idx = X.columns.get_loc(name)
        weight_path = []
        for i, v in reg_path.items():
            weight_path.append(v[num])
        plt.plot(lambdas, weight_path, label=name)
    plt.title('Feature weights vs $\lambda$")
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Regularization Paths')
    plt.legend()
    plt.savefig('A6d.png')
   
if __name__ == "__main__":
    main()
