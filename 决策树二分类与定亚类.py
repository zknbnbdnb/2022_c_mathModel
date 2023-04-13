from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# ----------第2题二分类--------

df1 = pd.read_excel('D:/pytorch/2022c建模/分类.xlsx') # 读取已确定玻璃类型的数据

X_2, y_2 = df1.iloc[:, 7:21].values, df1.iloc[:, 4].values
y_2 = LabelEncoder().fit_transform(y_2) # 数据编码
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.3,random_state=2022) # 训练集测试集7：3分


clf1 = DecisionTreeClassifier().fit(X_2_train, y_2_train)
pred_2 = clf1.predict(X_2_test) # 对模型进行预测
print(classification_report(pred_2, y_2_test))
fn=['sio2','na2o','k2o','cao','mgo',
    'al2o3','fe2o3','cuo','pbo','bao',
    'p2o5','sro','sno2','so2'] # 特征名字即化学成分含量表
plot_tree(clf1, feature_names=fn)
plt.savefig('tree.png')
plt.show()

# ----------第2题亚类分类--------

df = pd.read_excel('D:/pytorch/2022c建模/kmeans结果.xlsx') # 读取kmeans结果

X, y = df.iloc[:, 8:22].values, df.iloc[:, 5].values
y = LabelEncoder().fit_transform(y) #数据编码
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2022) # 训练集测试集7：3分

clf = DecisionTreeClassifier().fit(X_train, y_train) # 训练模型
pred1 = clf.predict(X_test)
print(classification_report(pred1, y_test)) # 得到结果准确率一簇额

cn=['QB I', 'QB II', 'QB III', 'GJ I','GJ II']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
plot_tree(clf, feature_names = fn, class_names=cn, filled = True);
fig.savefig('tree2.png')
plt.show() # 画图

data = pd.read_excel('D:/pytorch/2022c建模/第三题结果.xlsx') # 预测第三题未分类的玻璃所属亚类
X = data.iloc[:, 2:16]
pred2 = clf.predict(X)
print(pred2)




