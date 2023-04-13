import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
from scipy.stats import spearmanr, ttest_rel

# ******************灰色关联分析******************
from sklearn.preprocessing import StandardScaler


def dimensionlessProcessing(df_values, df_columns):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    res = scaler.fit_transform(df_values)
    return pd.DataFrame(res, columns=df_columns)


# 求第一列(影响因素)和其它所有列(影响因素)的灰色关联值
def GRA_ONE(data, m=0):  # m为参考列
    # 标准化
    data = dimensionlessProcessing(data.values, data.columns)
    # 参考数列
    std = data.iloc[:, m]
    # 比较数列
    ce = data.copy()

    n = ce.shape[0]
    m = ce.shape[1]

    # 与参考数列比较，相减
    grap = np.zeros([n, m])
    for i in range(m):
        for j in range(n):
            grap[j, i] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中的最大值和最小值
    mmax = np.amax(grap)
    mmin = np.amin(grap)
    ρ = 0.1  # 灰色分辨系数

    # 计算值
    grap = pd.DataFrame(grap).applymap(lambda x: (mmin + ρ * mmax) / (x + ρ * mmax))

    # 求均值，得到灰色关联值
    RT = grap.mean(axis=0)
    return pd.Series(RT)


# 调用GRA_ONE，求得所有因素之间的灰色关联值
def GRA(data):
    list_columns = np.arange(data.shape[1])
    df_local = pd.DataFrame(columns=list_columns)
    for i in np.arange(data.shape[1]):
        df_local.iloc[:, i] = GRA_ONE(data, m=i)
    return df_local

def ShowGRAHeatMap(data):
    # 色彩集
    colormap = plt.cm.RdBu
    plt.figure(figsize=(18,16))
    plt.title('GRA',y=1.05,size=18)

    draw = sns.heatmap(data.astype(float),linewidths=0.1,vmax=1.0,square=True,\
                cmap=colormap,linecolor='white',annot=True)
    plt.show()
    draw.get_figure().savefig('1.png')

# ---------灰色关联铅钡--------------

df = pd.read_excel('D:/pytorch/2022c建模/汇总信息表.xlsx', sheet_name='铅钡含严重风化')
cols = df.columns.values[7:21]
huise = GRA(df.iloc[:, 7:21]) # 得出灰色关联分析表

data = df.iloc[:, 7:21].values
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=cols)

ShowGRAHeatMap(huise)  # 用热力图可视化灰色关联分析表

# ---------灰色关联高钾--------------

df1 = pd.read_excel('D:/pytorch/2022c建模/汇总信息表.xlsx', sheet_name='高钾')
cols1 = df1.columns.values[7:21]
huise1 = GRA(df1.iloc[:, 7:21]) # 得出灰色关联分析表

data1 = df1.iloc[:, 7:21].values
scaler1 = StandardScaler()
data1 = scaler1.fit_transform(data1)
data1 = pd.DataFrame(data1, columns=cols)

ShowGRAHeatMap(huise1)  # 用热力图可视化灰色关联分析表

# ---------斯皮尔曼相关系数-铅钡--------------
data2 = data.values
res = []
res_p = []
for i in range(len(data2[0])):
    tmp1 = data2[:,i]
    tmp = []
    tmp_p = []
    for j in range(len(data2[0])):
        tmp2 = data2[:,j]
        tmp.append(spearmanr(tmp1, tmp2)[0]) # 对每个变量与其他变量求斯皮尔曼系数
        tmp_p.append(spearmanr(tmp1, tmp2)[1]) # 对每个变量与其他变量求斯皮尔曼系数的p值
    res.append(tmp)
    res_p.append(tmp_p)
res = np.asarray(res)
res_p = np.asarray(res_p)

colormap = plt.cm.RdBu
plt.figure(figsize=(18,16))
plt.title('spearmanr_qb',y=1.05,size=18)

draw = sns.heatmap(res.astype(float),linewidths=0.1,vmax=1.0,square=True,\
            cmap=colormap,linecolor='white',annot=True)
plt.show()
draw.get_figure().savefig('spearmanr_qb.png') # 斯皮尔曼系数热力图可视化

# ---------斯皮尔曼相关系数-高钾--------------

data3 = data1.values
res = []
res_p = []
for i in range(len(data3[0])):
    tmp1 = data3[:,i]
    tmp = []
    tmp_p = []
    for j in range(len(data3[0])):
        tmp2 = data3[:,j]
        tmp.append(spearmanr(tmp1, tmp2)[0])# 对每个变量与其他变量求斯皮尔曼系数
        tmp_p.append(spearmanr(tmp1, tmp2)[1]) # 对每个变量与其他变量求斯皮尔曼系数的p值
    res.append(tmp)
    res_p.append(tmp_p)
res = np.asarray(res)
res_p = np.asarray(res_p)

colormap = plt.cm.RdBu
plt.figure(figsize=(18,16))
plt.title('spearmanr_gj',y=1.05,size=18)

draw = sns.heatmap(res.astype(float),linewidths=0.1,vmax=1.0,square=True,\
            cmap=colormap,linecolor='white',annot=True)
plt.show()
draw.get_figure().savefig('spearmanr_gj.png') # 斯皮尔曼系数热力图可视化