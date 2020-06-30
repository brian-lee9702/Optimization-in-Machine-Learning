# Template Code for Assignment 1
 
# Your Name:      

# Import Packages

import pandas as pd
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

import cvxpy as cvx

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Lars


# ### Load Data

data = pd.read_csv('SpotifyFeaturesNormalized.csv')
data.head()


# ### Split Data

X = data.iloc[:,4:]
y = data.popularity

# do not change the random_state

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
N, p = X_train_full.shape
print('Number of features:',p)
print('Size of training set:',N)
print('Size of test set:',X_test.shape[0])


# # 1. Least Squares Regression (LSR)
# 
# ## Least Squares Regression (LSR)
# 
# Given a design matrix $\mathbf{X} \in \mathbb{R}^{N\times p}$, LSR seeks to minimize the _squared_ distance between it and the response vector $\mathbf{y}\in \mathbb{R}^{N}$. In other words, LSR attempts to find the hyperplane parameterized by $\mathbf{\beta}$ that best fit the exogenous variables in the design matrix $\mathbf{X}$ to the endogenous observations $\mathbf{y}$ by solving the following optimization $$\mathbf{\beta}^* \in \text{argmin}\{ \frac{1}{2}\| \mathbf{y-X\beta}\|_2^2\}$$
# For this assignment, we'll be using the `CVXPY` package to code the mathematical programs. [`CVXPY`](https://www.cvxpy.org/index.html) is a Python-embedded modeling language for convex optimization problems. Using its [atom functions](https://www.cvxpy.org/tutorial/functions/index.html#functions) it allows you to describe the problems in a natural way rather than use matrix reformulations.

# **_a) Complete the function `LSR_loss_fn` that constructs the objective function used to solve the above optimization problem:_**

def LSR_loss_fn(X, Y, beta):
    '''
           Returns the LSR objective function value

           Input:
                - X: (n x p) design matrix
                - Y: (n x 1) target vector
                - beta: (p x 1) coefficient vector
           Output:
                - z: (float) LSR objective function value
    '''
        loss_fn = 0.5 * cvx.sum_squares(Y.to_numpy() - X.to_numpy() @ beta)
        return loss_fn


# **_b) Using the entire training dataset, `X_train_full`, solve the LSR problem using the function from part a). Report:_**
# **_I. The optimal coefficients_**
# **_II. The optimal objective function value_**
# **_III. The solution time_**
# 


##### YOUR CODE HERE #####

start_time = time.time()
beta = cvx.Variable(12)
objective = cvx.Minimize(LSR_loss_fn(X_train_full, y_train_full, beta))
prob = cvx.Problem(objective)
prob.solve()
optimal_beta = beta.value
optimal_value = prob.value
solution_time = time.time() - start_time

print("Optimal Coefficients: ", optimal_beta)
print("\nOptimal Value: ", optimal_value)
print("\nSolution Time: ", solution_time)

# Answer:

# Alternatively, LSR can be solved using the normal equation: $$\beta^*=\mathbf{(X^TX)^{-1}X^Ty}$$ 
# 
# **_c) Complete the function `LSR_loss_fn` using `NumPy` notation that returns the solution to the normal equation._**



def LSR_norm_eq(X, Y):
    '''
        Returns the optimal regression coeffecients

        Input:
             - X: (n x p) design matrix
             - Y: (n x 1) target vector
        Output:
             - beta: (p x 1) valid coefficient vector
    '''
        Y = Y.to_numpy()
        X = X.to_numpy()
        XT = np.transpose(X)
        x_inv = np.linalg.inv(np.dot(XT,X))
        XTy =  np.dot(XT,Y)
        beta_hat = np.dot(x_inv,XTy)
        return beta_hat


# **_d) Using the entire training dataset, `X_train_full`, solve the LSR problem using the normal equation from part c). Report:_**
# 
# **_I. The optimal coefficients_**
# 
# **_II. The solution time_**
# 


##### YOUR CODE HERE #####

start_time = time.time()
opt_beta_hat = LSR_norm_eq(X_train_full, y_train_full)
end_time = time.time()-start_time

print("Optimal coefficients: ", opt_beta_hat)
print("Solution time: ", end_time)

# Answer:



# # 2. Least Absolute Shrinkage & Selection Operator (LASSO)
# 
# 
# LASSO is a feature selection method that constrains the regression problem by limiting its feasible region to the $\ell_1$-norm of the regression coefficients $\mathbf{\beta}$, represented above by the blue polyhedron.
# 
# **_a) Complete the function `L1_regularizer` that constructs the regularizing penalty term to be added to the regression problem's objective function._**


def L1_regularizer(beta):
    '''
        Returns the optimal regression coeffecients

        Input:
             - beta: (p x 1) valid coefficient vector
        Output:
             - l1 norm of vector
    '''
    return cvx.norm(beta, 1)


# Often in practice, as well as in this assignment, the LASSO problem is written and implemented in its Lagrangian form $$\underset{\mathbf{\beta}}{\min}\quad f_{q,\lambda}(\mathbf{\beta})=\frac{1}{q}\|\mathbf{y-X\beta}\|_q^q + \lambda \|\mathbf{\beta}\|_1$$ where $\lambda \in \mathbb{R}_+$ is a penalty hyperparameter that is set by the user and must be tuned.
# 
# **_b) Complete the functions `LSR_L1_obj_fn` and `LAD_L1_obj_fn` that add the penalty term to construct the full objective functions used for the LASSO regression problems._**



def LSR_L1_obj_fn(X, Y, beta, lambd):
    '''
        Returns the LSR-LASSO objective function value

        Input:
             - X: (n x p) design matrix
             - Y: (n x 1) target vector
             - beta: (p x 1) coefficient vector
             - lambd: (float) hyper-parameter
        Output:
             - z: (float) LSR-LASSO objective function value
    '''
    return LSR_loss_fn(X,Y,beta)+lambd*L1_regularizer(beta)

# Since we're trying out multiple values for our model's hyper parameters ($\lambda$ in the case of LASSO), we need to use a new data-set for each new value in order to avoid overfitting. These data-sets are known as validation sets. There are many different validation frameworks but we'll implement [k-fold cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) to test out k values for our hyperparameter.
# 
# **_c) Generate 10 linearly spaced $\lambda \in [300,1000]$. For each $\lambda_j, j = 0,\dots,9$, optimize $f_2(\beta,\lambda_j)$. Due to numerical issues, round the values of $\beta$ to the 7th significant digit. Define $\lambda$ as a [`CVXPY Parameter`](https://www.cvxpy.org/api_reference/cvxpy.expressions.html#parameter). For each $\lambda_j$, report:_**
# 
# **_I. The coefficients_**
# 
# **_II. The number of non-zero coefficients in the model_**
# 
# **_III. The validation $R^2$ values the model_**
# 
# **_Which set of coefficients should we use out of sample?_**
# 
# _Note: The code for the k-fold crossvalidation has already been implemented for you._



kf_lasso = KFold(n_splits=10, shuffle=True, random_state=0)

##### YOUR CODE HERE #####
ran = np.linspace(300, 1000, 10)
lambd =  cvx.Parameter(nonneg=True)
opt_betas = []
nonzero_count = []
for lam in ran:
    lambd.value = lam
    beta = cvx.Variable(12)
    objective = cvx.Minimize(LSR_L1_obj_fn(X_train_full, y_train_full, beta, lambd))
    prob = cvx.Problem(objective)
    prob.solve()
    count = 0
    for param in beta.value:
        if np.abs(param) > 10 ** -15:
            count += 1
    nonzero_count.append(count)
    opt_betas.append(beta.value)


R2_vals_list = []
lam_ct = 0
o_tracker = 1
for train_index, test_index in kf_lasso.split(X_train_full):
    X_fold_t = X_train_full.iloc[train_index,:]
    y_fold_t = y_train_full.iloc[train_index]
    X_fold_v = X_train_full.iloc[test_index,:]
    y_fold_v = y_train_full.iloc[test_index]
    R2_vals = []
    i_tracker = 1
    for lam in ran:
        lambd.value = lam
        beta = cvx.Variable(12)
        objective = cvx.Minimize(LSR_L1_obj_fn(X_fold_t, y_fold_t, beta, lambd))
        prob = cvx.Problem(objective)
        prob.solve()
        R2_vals.append(r2_score(y_fold_v, X_fold_v.dot(beta.value)))
        lam_ct += 1
        print("Inside: ", i_tracker)
        i_tracker+=1
    R2_vals_list.append(R2_vals)
    print("Outside: ", o_tracker)
    o_tracker +=1





    
##### YOUR CODE HERE #####



# Answer:

# One of the problems with LASSO is that it is difficult to tune its hyperparameter. First, it is a continuous value, and thus may take on a theoretically infinite number of values. Further, the same value for $\lambda$, used in the same model on two different datasets, may yield different results. 
# 
# **_d) Run $f_1(\beta,\lambda_4)$ using the set of $\lambda$s from c) on the 10 training sets generated by our k-fold framework. How many non-zero coefficients does each data-set have in its model? Due to numerical issues, round the values of $\beta$ to the 7th significant digit._**



##### YOUR CODE HERE #####

def L1_Lasso(X, Y, beta, lambd):
    '''
        Returns the LSR-LASSO objective function value

        Input:
             - X: (n x p) design matrix
             - Y: (n x 1) target vector
             - beta: (p x 1) coefficient vector
             - lambd: (float) hyper-parameter
        Output:
             - z: (float) LSR-LASSO objective function value
    '''
    exp = Y.to_numpy() - X.to_numpy() @ beta
    return cvx.norm(exp, 1) + lambd * cvx.norm(beta, 1)

opt_betas_d = []
nonzero_count_d = []
for train_index, test_index in kf_lasso.split(X_train_full):
    X_fold_t = X_train_full.iloc[train_index,:]
    y_fold_t = y_train_full.iloc[train_index]
    X_fold_v = X_train_full.iloc[test_index,:]
    y_fold_v = y_train_full.iloc[test_index]
    beta = cvx.Variable(12)
    objective = cvx.Minimize(L1_Lasso(X_fold_t, y_fold_t, beta, ran[3]))
    prob = cvx.Problem(objective)
    prob.solve()
    count = 0
    for param in beta.value:
        if np.abs(param) > 10 ** -15:
            count += 1
    nonzero_count.append(count)
    opt_betas.append(beta.value)
    
##### YOUR CODE HERE #####


# Answer:

# **_e) Generate 10 equally spaced $\lambda \in [300,1000]$. For each $\lambda_j, j = 0,\dots,9$, optimize $f_1(\beta,\lambda_j)$. Due to numerical issues, round the values of $\beta$ to the 7th significant digit. For each $\lambda_j$, report:_**
# 
# **_I. The coefficients_**
# 
# **_II. The number of non-zero coefficients in the model_**
# 
# **_III. The validation $R^2$ values the model_**
# 
# **_Which set of coefficients should we use out of sample?_**
# 
# 


##### YOUR CODE HERE #####

for train_index, test_index in kf_lasso.split(X_train_full):
    X_fold_t = X_train_full.iloc[train_index,:]
    y_fold_t = y_train_full.iloc[train_index]
    X_fold_v = X_train_full.iloc[test_index,:]
    y_fold_v = y_train_full.iloc[test_index]
    
##### YOUR CODE HERE #####


# Answer:

new_R2_list = []
for i in range(10):
    hold_list = []
    for sublist in R2_vals_list:
        hold_list.append(sublist[i])
    new_R2_list.append(hold_list)

R2_means = []
for i in range(10):
    avg_over_folds = np.mean(new_R2_list[i])
    R2_means.append(avg_over_folds)