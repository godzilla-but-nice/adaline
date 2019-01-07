import pandas as pd
import numpy as np
from AdalineGD import *
import matplotlib.pyplot as plot
from plot_decision_regions import *
import pdb

df = pd.read_csv('training_data/iris.csv', header = None)

# Extract sepal and petal length
X = df.iloc[0:100, [0, 2]].values
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# get labels
y = df.iloc[0:100, 4].values
Y = np.where(y == 'Iris-setosa', -1, 1)

# set up figure for cost function
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Fit adaline to iris data
eta1 = 0.0001
eta2 = 0.01
ada1 = AdalineGD(eta = eta1, n_iter = 20).fit(X, Y)
ada2 = AdalineGD(eta = eta2, n_iter = 20).fit(X_std, Y)

# titles for my figure
ax[0].set_title('Raw Input (eta = {0})'.format(eta1))
ax[1].set_title('Standardized Input (eta = {0})'.format(eta2))

# plot learning using cost function
ax[0].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker = 'o')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Sum-squared-error')
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Sum-squared-error')

plt.savefig('irisBGD_cost.png')

# plot decision regions for unstandardized adaline
plt.figure()
plot_decision_regions(X, Y, classifier=ada1)
plt.title('Adaline decision with unstandardized input')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='best')
plt.savefig('raw_iris_decision.png')

# plot decision regions for standardized adaline
plt.figure()
plot_decision_regions(X_std, Y, classifier=ada2)
plt.title('Adaline decision with standardized input')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='best')
plt.savefig('std_iris_decision.png')
