from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
print(faces.data.shape)

pca = PCA(n_components=150,whiten=True,random_state=42)#2000多维太高了本本速度不行，先用PCA降维
svc = SVC(kernel='rbf',class_weight='balanced')
model = make_pipeline(pca,svc)

Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)

para_grid = {'svc__C':[1,5,10],
             'svc__gamma':[0.0001,0.0005,0.001]}

grid = GridSearchCV(model,para_grid)#交叉验证
grid.fit(Xtrain,ytrain)
print(grid.best_params_)#看下最好的参数

best_model = grid.best_estimator_

y_predict = best_model.predict(Xtest)

print(classification_report(ytest, y_predict,
                            target_names=faces.target_names))

mat = confusion_matrix(ytest, y_predict)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('confusion_matrix')



