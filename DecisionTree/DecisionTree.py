import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.grid_search import GridSearchCV
housing = fetch_california_housing()
print(housing.DESCR)


#普通的回归树
data_train, data_test, target_train, target_test = \
    train_test_split(housing.data, housing.target, test_size = 0.1, random_state = 42)
dtr = tree.DecisionTreeRegressor(random_state = 42)
dtr.fit(data_train, target_train)

print(dtr.score(data_test, target_test))


#用GridSearchCV试试参数调优
tree_param_grid = { 'min_samples_split': list((3,6,9))}
grid = GridSearchCV(tree.DecisionTreeRegressor(),param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print(grid.grid_scores_)
print(grid.best_params_, grid.best_score_)