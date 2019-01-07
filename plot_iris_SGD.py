import pandas as pd
import numpy as np
from AdalineSGD import *
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
fig, ax = plt.subplots()

# Fit adaline to iris data
eta = 0.01
ada = AdalineSGD(eta = eta, n_iter = 15).fit(X_std, Y)

# titles for my figure
plt.title('Standardized Input (eta = {0})'.format(eta))

# plot learning using cost function
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epoch')
plt.ylabel('Sum-squared-error')

plt.savefig('plots/iris_SGD_cost.png')

# plot decision regions for unstandardized adaline
plt.figure()
plot_decision_regions(X_std, Y, classifier=ada)
plt.title('Adaline decision with standardized input')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='best')
plt.savefig('plots/std_iris_decision_SGD.png')
