from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt



iris = load_iris()
X_set, y_set = load_iris(return_X_y=True)

X = iris.data[:, 2:]
y = iris.target


clf = DecisionTreeClassifier(random_state=1)
clf = clf.fit(X_set, y_set)

plot_tree(clf)
plt.show()