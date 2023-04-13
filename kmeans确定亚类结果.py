import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster._hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------第二问聚类--------

dataz = pd.read_excel('D:/pytorch/2022c建模/汇总信息表.xlsx', sheet_name='高钾')
data = pd.read_excel('D:/pytorch/2022c建模/汇总信息表.xlsx', sheet_name='铅钡含严重风化')
# 分别读取两个不同的类别的玻璃树

data1 = data.iloc[:, 7:21].values
data2 = dataz.iloc[:, 7:21].values

min_size = min(data1.shape[1], data2.shape[1]) # 得出来最小数据量，当k值最大值

sil_score_q = [] # 轮廓系数列表
inert_q = [] # SSE列表
for i in range(2, min_size):
    kmeans = KMeans(n_clusters=i, n_init=100,max_iter=1000).fit(data1)
    sil_score_q.append(silhouette_score(data1, kmeans.labels_)) # 求轮廓系数
    inert_q.append(kmeans.inertia_) # 求SSE
plt.figure(figsize=(6, 4))
plt.plot(range(2, min_size), sil_score_q, 'o-')
plt.xlabel('k', fontsize=20)
plt.ylabel('silhouette score', fontsize=20) # 确定x，y的标签以及字体大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)# 确定x，y轴大小
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.savefig('qbS-Score.png')
plt.show() # 作图

plt.figure(figsize=(6, 4))
plt.plot(range(2, min_size), inert_q, 'o-')
plt.xlabel('k', fontsize=20)
plt.ylabel('SSE', fontsize=20)# 确定x，y的标签以及字体大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)# 确定x，y轴大小
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.savefig('qbSSE.png')
plt.show()# 作图

sil_score_k = []# 轮廓系数列表
inert_k = []# SSE列表
for i in range(2, min_size):
    kmeans = KMeans(n_clusters=i, n_init=100,max_iter=1000).fit(data2)
    sil_score_k.append(silhouette_score(data2, kmeans.labels_))# 求轮廓系数
    inert_k.append(kmeans.inertia_)# 求SSE

plt.figure(figsize=(6, 4))
plt.plot(range(2, min_size), sil_score_k, 'o-')
plt.xlabel('k', fontsize=20)
plt.ylabel('silhouette score', fontsize=20)# 确定x，y的标签以及字体大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15) # 确定x，y轴大小
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.savefig('gjS-Score.png')
plt.show()# 作图

plt.figure(figsize=(6, 4))
plt.plot(range(2, min_size), inert_k, 'o-')
plt.xlabel('k', fontsize=20)
plt.ylabel('SSE', fontsize=20)# 确定x，y的标签以及字体大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)# 确定x，y轴大小
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.savefig('gjSSE.png')
plt.show()# 作图

# 确定好了k后开始确定聚类中心

qinaBa = KMeans(n_clusters=3, n_init=100, max_iter=1000).fit(data1)
gaoK = KMeans(n_clusters=2, n_init=100, max_iter=1000).fit(data2)

pred_index_q = qinaBa.fit_predict(data1)
pred_index_k = gaoK.fit_predict(data2)

print(pred_index_q, pred_index_k) # 每个玻璃的亚类划分
print(qinaBa.cluster_centers_, gaoK.cluster_centers_) # 每个亚类的聚类中心

