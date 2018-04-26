import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

titanic = pd.read_csv("titanic_train.csv")
print (titanic.describe())

#预处理
titanic['Age'] = titanic['Age'].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

#训练，交叉验证
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
scores = model_selection.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())

#看一下特征重要性
selector = SelectKBest(f_classif, k=4)
selector.fit(titanic[predictors], titanic["Survived"])
scores = -np.log10(selector.pvalues_)

#特征重要性画图
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()