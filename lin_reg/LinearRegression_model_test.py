import numpy as np
# Test data
from sklearn import datasets
from lin_reg.LRegression import LinReg
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=10, random_state=40)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)
y = y.reshape(1000, 1)


manual_clf = LinReg()
manual_clf.fit(X, y)
print("Weights - {} and Bias - {}".format(manual_clf.weights, manual_clf.bias[0]))

# skleran LinearRegression model
clf = LinearRegression()
clf.fit(X, y)
print(clf.score(X, y), clf.coef_, clf.intercept_)

# Plots
fig = plt.figure(figsize=(15,5))

ax1 = plt.subplot(1, 2, 1)
plt.plot(manual_clf.cost_history)
plt.title("Error rate")
plt.xlabel('Training iter')
plt.ylabel('MSE')

ax2 = plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue')
x = np.linspace(-5, 6, 1000)
y = (manual_clf.weights*x + manual_clf.bias[0]).T
plt.plot(x, y)
plt.title("Linear Regression line")
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()