from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# 第3问确定类型

df = pd.read_excel('D:/pytorch/2022c建模/kmeans结果.xlsx') # 导入kmeans的结果

X, y = df.iloc[:, 8:22].values, df.iloc[:, 4].values
y = LabelEncoder().fit_transform(y) # 将定类变量转为定量变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022) # 训练集测试集7：3分


clf = svm.SVC(C=1.0, kernel='linear').fit(X_train, y_train) # 采用正则化系数=1，且核函数为线性核进行训练预测
pred1 = clf.predict(X_test)
print(classification_report(pred1, y_test)) # 预测模型的准确率以确定模型

new_data = pd.read_excel('D:/pytorch/2022c建模/第三题结果.xlsx', sheet_name='总表') #读取未分类玻璃的数据
need_pred_X = new_data.iloc[:, 2:16].values
clf_3 = svm.SVC(C=1.0, kernel='linear').fit(X, y)# 使用前面所有数据进行训练, 确定未知玻璃的类型
base_pred = clf_3.predict(need_pred_X)
print(base_pred) # 得出数据结果

c_list = [1e2, 1e1, 1, 1e-1, 1e-2, 1e-3] # 超参数： 正则化系数列表
method_list = ['rbf', 'linear', 'poly'] # 超参数：核函数列表
res = []
x_label = []

for i in range(len(c_list)):
    for j in range(len(method_list)):
        x_label.append(str(c_list[i]) + ',' + method_list[j])
        clf_tmp = svm.SVC(C=c_list[i], kernel=method_list[j])
        clf_tmp.fit(X, y)
        pred_tmp = clf_tmp.predict(need_pred_X)
        scoure = 0
        if pred_tmp.all() == base_pred.all():
            res.append(1)
        else:
            res.append(0) # 计算不同超参数组合的准确率来验证敏感性


plt.figure(figsize=(6,6))
plt.plot(x_label, res, 'o-')
plt.ylabel('ACC', fontsize=20)
plt.xlabel('Hyper parameters', fontsize=20)
plt.axhline(1, color='r')
plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=18)
plt.subplots_adjust(bottom=0.3)
plt.savefig('svm.jpg')
plt.show() # 作图说明




