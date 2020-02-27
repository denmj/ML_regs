import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline
from sklearn.model_selection import GridSearchCV

X_set, y_set = load_breast_cancer(return_X_y=True)

print("X_set shape: {}".format(X_set.shape))
print("y_set shape {}".format(y_set.shape))

X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, random_state=42)

svc_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('svm_clf', SVC())]
    )
# print(svc_pipeline.get_params().keys())
# Set the parameters by cross-validation
tuned_parameters = {'svm_clf__gamma': [1e-3, 1e-4],
                     'svm_clf__C': [1, 10, 100, 1000]}

search = GridSearchCV(svc_pipeline, tuned_parameters, n_jobs=-1)
search.fit(X_train, y_train)

print(search.best_estimator_)