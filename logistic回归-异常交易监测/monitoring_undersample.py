import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split,KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score,accuracy_score
import itertools

#交叉验证找最佳的正则化参数
def Kfold_scores(x_train_data, y_train_data):
    # 分成5份做交叉验证，这里只返回了最终每个组合的索引，如第一组可能为  训练：份1+份2+份3+份4，测试：份5
    fold = KFold(len(y_train_data), 5, shuffle=False)

    # 正则化参数选几个，一会逐个试一下哪个好
    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'],
                                 dtype='float64')
    # results_table['Mean recall score'] = results_table['Mean recall score'].astype('float64')
    results_table['C_parameter'] = c_param_range

    # 每个fold有两个list: 训练list的索引train_indices = indices[0], 测试list的索引test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        accs = []
        for iteration, indices in enumerate(fold, start=1):
            # 生成一个分类器
            lr = LogisticRegression(C=c_param, penalty='l1')

            # 开始训练
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            # 用测试集预测
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :])

            # 计算 recall score
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration, ': recall score = ', recall_acc)

            # 计算 精度
            acc = accuracy_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            accs.append(acc)
            print('Iteration ', iteration, ': accuracy_score = ', acc)

        # 5组求平均.
        results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

        # 5组求平均.
        results_table.ix[j, 'Mean accuracy score'] = np.mean(accs)
        j += 1
        print('')
        print('Mean accuracy score ', np.mean(accs))
        print('')

    best_para_c = results_table.loc[results_table['Mean recall score'].idxmax(), 'C_parameter']  # 先用recall做指标看一下

    # 选出最好的正则化参数
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_para_c)
    print('*********************************************************************************')

    return best_para_c

#画混淆矩阵
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data  =pd.read_csv('creditcard.csv')
data['standAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))#数据标准化
data = data.drop(['Time','Amount'],axis=1)

X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']

# 把正常和异常数据行的索引都取出来
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index

# 下采样，在正常样本里取出和异常样本同样数目的行索引
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# 把索引拼起来
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# 利用索引拼数据
under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# 切分全部数据集，可以测试用
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# 切分下采样数据集
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
best_c = Kfold_scores(X_train_undersample,y_train_undersample)


#在整个数据集上做下测试
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

