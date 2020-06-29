# Optimization in Machine Learning (Winter 2020)
# Assignment 2
# Template Code


# Before you start, please read the instructions of this assignment
# For any questions, please email to yhe@mie.utoronto.ca
# For free-response parts, please submit a seperate .pdf file

# Your Name:
# Email:

"""
Problem 1: Linear Support Vector Machine
"""

# Import Libraries
from numpy import *
import numpy as np
import pandas as pd
import cvxpy as cp
import time
import cvxopt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pylab as plt
import matplotlib

# Import Data
data1 = pd.read_csv('prob1data.csv',header=None).values
X = data1[:,0:2]
y = data1[:,-1]

for i in range(len(y)):
  if y[i]==0:
    y[i]=-1
# Hint: examine the data before you start coding SVM

# Problem (1a)
def LinearSVM_Primal (X, y, C):
  start_time = time.time()
  w = cp.Variable(2)
  b = cp.Variable()
  xi = cp.Variable(51)
  obj = cp.Minimize(1 / 2 * cp.sum_squares(w) + C*cp.sum(xi))
  constraints1 = [cp.multiply(y, (w.T * X.T + b)) >= np.ones(51) - xi]
  constraints2 = [xi >= np.zeros(51)]
  constraints = constraints1 + constraints2
  prob = cp.Problem(obj, constraints)
  prob.solve()
  sol_time = time.time() - start_time

  return w, b, sol_time

# Compute the decision boundary
# -------- INSERT YOUR CODE HERE -------- # 
SVM1 = LinearSVM_Primal(X,y,1)
w1 = SVM1[0].value
b1 = SVM1[1].value
sol_time = SVM1[2]
print("The optimal w is ", w1)
print("The optimal b is ", b1)
print("The solution time is ", sol_time)
# Compute the optimal support vectors
# -------- INSERT YOUR CODE HERE -------- #
residues = []
supporting = []
for i in range(len(X)):
  res = w1[0]*X[i][0]+w1[1]*X[i][1]+b1
  residues.append(res)
  if np.abs(1-res) < 0.05:
    supporting.append(i)
  elif np.abs(-1-res) < 0.05:
    supporting.append(i)
support_vectors = [X[i] for i in supporting]
print("The support vectors are ", support_vectors, " with labels ", supporting)

# Problem (1b)

def LinearSVM_Dual (X, y, C):
  start_time = time.time()
  n, p = X.shape
  y = y.reshape(-1,1) * 1.
  X_dash = y * X
  # Complete the following code:
  # cvxopt_solvers.qp(P, q, G, h, A, b)
  # objective:    (1/2) x^T P x + q^T x
  # constraints:  Gx < h
  #               Ax = b
  # example could be found here:
  # https://cvxopt.org/userguide/coneprog.html#quadratic-programming

  P = cvxopt_matrix(X_dash.dot(X_dash.T))
  q = cvxopt_matrix(-np.ones((n,1)))
  G = cvxopt_matrix(np.vstack((-np.diag(np.ones(n)), np.identity(n))))
  h = cvxopt_matrix(np.hstack((np.zeros(n), np.ones(n) * C)))
  A = cvxopt_matrix(y.reshape(1,-1))
  b = cvxopt_matrix(np.zeros(1))

  cvxopt_solvers.options['show_progress'] = False
  cvxopt_solvers.options['abstol'] = 1e-10
  cvxopt_solvers.options['reltol'] = 1e-10
  cvxopt_solvers.options['feastol'] = 1e-10

  # -- INSERT YOUR CODE HERE -- #
  sol = cvxopt_solvers.qp(P, q, G, h, A, b)
  # -- INSERT YOUR CODE HERE -- #
  alphas = np.array(sol['x'])
  sol_time = time.time() - start_time

  return alphas, sol_time


# Compute the decision boundary
# -------- INSERT YOUR CODE HERE -------- #
alphas, sol_time = LinearSVM_Dual(X,y,1)
y_re = y.reshape(-1,1) * 1.
w2 = np.sum(alphas * y_re * X, axis=0)
cond = (alphas > 1e-6).reshape(-1)
b2 = y[cond] - np.dot(X[cond], w2)
b2 = np.mean(b2)
alphas_p = alphas.tolist()
for i in range(51):
   alphas_p[i]=alphas_p[i][0]
print("The optimal alpha is ", alphas_p)
print("The optimal w is ", w2)
print("The optimal b is ", b2)
print("The solution time is ", sol_time)





# Compute the optimal support vectors
# -------- INSERT YOUR CODE HERE -------- #
residues = []
supporting = []
for i in range(len(X)):
  res = w2[0]*X[i][0]+w2[1]*X[i][1]+b2
  residues.append(res)
  if np.abs(1-res) < 0.1:
    supporting.append(i)
  elif np.abs(-1-res) < 0.1:
    supporting.append(i)
support_vectors = [X[i] for i in supporting]
print("The support vectors are ", support_vectors, " with labels ", supporting)
# Problem (1d)
def Linearly_separable (X, y):
  w = cp.Variable(2)
  b = cp.Variable()
  t = cp.Variable()

  obj = cp.Maximize(t)
  constraints1 = [y[i]*(w.T * X[i] - b) >= 1 for i in range(len(X))]

  constraints = constraints1 + [cp.norm(w,1) <= 1]

  prob = cp.Problem(obj, constraints)
  prob.solve()

  if prob.status == 'infeasible':
    sep = 0
  else:
    sep = 1

  return sep

# Problem (1f)

def l2_norm_LinearSVM_Primal (X, y, C):
  start_time = time.time()
  w = cp.Variable(2)
  b = cp.Variable()
  xi = cp.Variable(51)
  obj = cp.Minimize(1 / 2 * cp.sum_squares(w) + C/2 * cp.sum_squares(xi))
  constraints = [cp.multiply(y, (w.T * X.T + b)) >= np.ones(51) - xi]
  prob = cp.Problem(obj, constraints)
  prob.solve()
  sol_time = time.time() - start_time
  return w, b, sol_time

# Compute the decision boundary
# -------- INSERT YOUR CODE HERE -------- # 
SVM3 = l2_norm_LinearSVM_Primal(X, y, 1)
w3 = SVM3[0].value
b3 = SVM3[1].value
sol_time = SVM3[2]
print("The optimal w is ", w3)
print("The optimal b is ", b3)
print("The solution time is ", sol_time)

# Compute the optimal support vectors
# -------- INSERT YOUR CODE HERE -------- #
residues = []
supporting = []
for i in range(len(X)):
  res = w3[0]*X[i][0]+w3[1]*X[i][1]+b3
  residues.append(res)
  if np.abs(1-res) < 0.05:
    supporting.append(i)
  elif np.abs(-1-res) < 0.05:
    supporting.append(i)
support_vectors = [X[i] for i in supporting]
print("The support vectors are ", support_vectors, " with labels ", supporting)

# Problem (1g)

def l2_norm_LinearSVM_Dual (X, y, C):
  zero_tol = 1e-7
  start_time = time.time()
  n, p = X.shape
  y = y.reshape(-1, 1) * 1.
  X_dash = y * X

  cvxopt_solvers.options['show_progress'] = False
  cvxopt_solvers.options['abstol'] = 1e-10
  cvxopt_solvers.options['reltol'] = 1e-10
  cvxopt_solvers.options['feastol'] = 1e-10
  # objective:    (1/2) x^T P x + q^T x
  # constraints:  Gx < h
  #               Ax = b
  # -------- INSERT YOUR CODE HERE -------- #
  P = cvxopt_matrix(X_dash.dot(X_dash.T)+1/C * np.identity(n))
  q = cvxopt_matrix(-np.ones((n, 1)))
  G = cvxopt_matrix(-np.identity(n))
  h = cvxopt_matrix(np.zeros(n))
  A = cvxopt_matrix(y.reshape(1, -1))
  b = cvxopt_matrix(np.zeros(1))

  cvxopt_solvers.options['show_progress'] = False
  cvxopt_solvers.options['abstol'] = 1e-10
  cvxopt_solvers.options['reltol'] = 1e-10
  cvxopt_solvers.options['feastol'] = 1e-10

  sol = cvxopt_solvers.qp(P, q, G, h, A, b)

  # -- INSERT YOUR CODE HERE -- #
  alphas = np.array(sol['x'])
  sol_time = time.time() - start_time

  return alphas, sol_time

# Compute the decision boundary
# -------- INSERT YOUR CODE HERE -------- # 
alphas, sol_time = l2_norm_LinearSVM_Dual(X,y,1)
y_re = y.reshape(-1,1) * 1.
w4 = np.sum(alphas * y_re * X, axis=0)
cond = (alphas > 1e-4).reshape(-1)
b4 = y[cond] - np.dot(X[cond], w4)
b4 = np.mean(b4)
print("The optimal alpha is ", alphas)
print("The optimal w is ", w4)
print("The optimal b is ", b4)
print("The solution time is ", sol_time)

# Compute the optimal support vectors
# -------- INSERT YOUR CODE HERE -------- #
residues = []
supporting = []
for i in range(len(X)):
  res = w4[0]*X[i][0]+w4[1]*X[i][1]+b4
  residues.append(res)
  if np.abs(1-res) < 0.05:
    supporting.append(i)
  elif np.abs(-1-res) < 0.05:
    supporting.append(i)
support_vectors = [X[i] for i in supporting]
print("The support vectors are ", support_vectors, " with labels ", supporting)

# Problem (1h)

# Plot the decision boundaries and datapoints
# -------- INSERT YOUR CODE HERE -------- #
x_val = np.linspace(0, 4, 10)
y_val = -(w2[0] * x_val + b2) / w2[1]
y_val2 = -(w3[0] * x_val + b3) / w3[1]
label = [1, -1]
colors = ['blue', 'red']
y_label = y.tolist()
plt.scatter(X[:,0],X[:,1], c=y_label, cmap=matplotlib.colors.ListedColormap(colors))
plt.plot(x_val,y_val, label="l1 SVM")
plt.plot(x_val,y_val2, label="l2 SVM")
plt.legend(loc="upper left")

"""
Problem 2: Kernal Support Vector Machine and Application
"""


# Import libraries
import numpy as np
from numpy import *
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
import sklearn
from mlxtend.plotting import plot_decision_regions
import matplotlib.pylab as plt
import matplotlib
from scipy.spatial.distance import pdist, squareform
import scipy

data2 = pd.read_csv('prob2data.csv',header=None).values
X = data2[:,0:2]
y = data2[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2020)
# Problem (2a)

def gaussian_kernel(sigma):
    def gaussian_kernel_sigma(x1, x2):
        K = np.zeros((shape(x1)[0], shape(x2)[0]))
        for i,x in enumerate(x1):
          for j,y in enumerate(x2):
            K[i,j] = np.exp(-np.linalg.norm(x-y)**2 / (2*sigma**2))
        return K
    return gaussian_kernel_sigma

g_ker = gaussian_kernel(0.1)
# Problem (2b)

kernel_SVM = SVC(kernel=g_ker, C=1)
kernel_SVM.fit(X_train,y_train)

# Compute # of optimal support vectors
n_support = kernel_SVM.n_support_
print("Number of optimal support vectors: ", n_support)

# Compute prediction error ratio in test set
test_pred = kernel_SVM.predict(X_test)
ct=0
for i in range(len(y_test)):
    if y_test[i] == test_pred[i]:
      ct+=1
accuracy = ct/len(y_test)
error = 1-accuracy
print("The error rate is: ", error)

# Plot the decision boundary with all datapoints
colors = ['blue', 'red']
y_label = y_train.tolist()
plt.scatter(X_train[:,0],X_train[:,1], c=y_label, cmap=matplotlib.colors.ListedColormap(colors))
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = kernel_SVM.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
           linestyles=['-'])

# Import data for (2c) - (2e)
data3 = pd.read_csv('votes.csv')
X = data3[['white','black','poverty','density','bachelor','highschool','age65plus','income','age18under','population2014']]
X = X.values
X = preprocessing.scale(X)

# Problem (2c)
y_set = data3[['clinton', 'trump']]
y_set = y_set.values
y = []
for i in range(len(y_set)):
  if y_set[i][0] < y_set[i][1]:
    y.append(0)
  else:
    y.append(1)

# Train / test split for (2d) - (2e)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 2020)

# Problem (2d)

# You may use SVC from sklearn.svm
for deg in np.arange(1,6,1):
  votes_SVM = SVC(kernel='poly', degree=deg)
  votes_SVM.fit(X_train, y_train)
  n_support = votes_SVM.n_support_
  print("Degree ", deg)
  print("Number of support vectors: ", n_support)

  test_pred = votes_SVM.predict(X_train)
  ct = 0
  for i in range(len(y_train)):
    if y_train[i] == test_pred[i]:
      ct += 1
  accuracy = ct / len(y_train)
  error = 1 - accuracy
  print("The training set error rate is: ", error)

  test_pred = votes_SVM.predict(X_test)
  ct = 0
  for i in range(len(y_test)):
    if y_test[i] == test_pred[i]:
      ct += 1
  accuracy = ct / len(y_test)
  error = 1 - accuracy
  print("The test set error rate is: ", error)
