from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X_, y_ = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=41)


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

vote_clf = VotingClassifier(
    estimators=[('lreg', log_clf),
                ('rand_clf', rnd_clf),
                ('svm', svm_clf)],
    voting='hard'
)

for clf in (log_clf, rnd_clf, svm_clf, vote_clf):
    clf.fit(X_train, y_train)
    scrs = clf.score(X_test, y_test)
    print(clf.__class__.__name__, scrs)


