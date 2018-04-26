import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import ensemble

housing = fetch_california_housing()
print(housing.DESCR)


#adaboost
data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)
dtr = ensemble.AdaBoostRegressor(random_state = 42)
dtr.fit(data_train, target_train)

print(dtr.score(data_train, target_train))
print(dtr.score(data_test, target_test))


#randomforest
regr = ensemble.RandomForestRegressor(random_state = 0)
regr.fit(data_train, target_train)

print(regr.score(data_test, target_test))
print(regr.feature_importances_)