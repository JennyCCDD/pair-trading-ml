# -*- coding: utf-8 -*-
"""
@author: Mengxuan Chen
@emails: chenmx19@tsinghua，mails.edu.cn
@description:
    # 配对交易
    reference: https://mp.weixin.qq.com/s/zfnq0ihXi6PYv0Czdw7ufA

@revise log:
    2022.04.07 创建程序
"""

import pandas as pd
import numpy as np
import os
import re
import datetime
import time
import copy
import warnings

warnings.filterwarnings('ignore')

#%%
close = pd.read_excel('./data/000300.SZclose20180307-20210407.xlsx',index_col=0)


#%% preprocessing
# 假设有n只股票，首先计算每只股票的标准化后的日度收益率，标准化是指用收益率除了标准差（滚动窗口，如120日Rolling Standard Deviation）。
def ret_preprocessing(CLOSE,window_=120):
    ret = CLOSE.pct_change(1)
    rolling_dev = ret.rolling(window = window_).std()
    ret_standard = ret / rolling_dev
    return ret_standard

ret = ret_preprocessing(close)

#%% PCA
# # 对以上收益率计算协方差矩阵，并进行PCA降维，选取前K个特征值，这样每个股票就有了K个特征。
from sklearn.decomposition import PCA

def PCA_de(DATA, features = 50):
    pca = PCA(n_components=features).fit(DATA.dropna())
    return pca.components_

features = PCA_de(ret)

#%% find pairs
# 非监督学习：利用以上特征结合聚类算法找到潜在的股票配对
# 非监督学习算法有很多，但满足当前应用场景的算法必须有以下特点：
# 无需指定聚类的数量;无需对所有股票都进行分类;对于异常值的处理;聚类的形状没有特定的假设
# https://www.cnblogs.com/pinard/p/6217852.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
def find_pairs(DATA, method='DBSCANE'):
    DATA = StandardScaler().fit_transform(DATA)
    clustering = DBSCAN(eps=6, min_samples=2).fit(DATA.T)
    labels = clustering.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return labels, n_clusters_,n_noise_


labels, n_clusters_, n_noise_ = find_pairs(features)
print(labels)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

#%% select pairs
# 配对选取：应用一系列准则对以上配对进行筛选
# 通过以上聚类，我们可以找到潜在的股票对（大部分情况下是一组股票），在筛选股票对的时候，作者采用了以下4个硬性指标：
# 1、两个股票必须协整
# 2、价差的Hurst指数要小于0.5（没有趋势性）
# 3、均值回归的半衰期（half-life of mean-reversion）要在合理区间内
# 4、历史上有足够多的交易机会
pairs = pd.concat([pd.Series(ret.columns), pd.Series(labels)],axis=1)
pairs.columns = ['code','label']
pairs = pairs[~(pairs['label'].isin([-1]))]
pairs = pairs.sort_values(by='label')

pairs_list = []
for i in range(len(pairs)-1):
    for j in range(1, len(pairs)):
        if pairs.iloc[i,1] == pairs.iloc[j,1]:
            pairs_list.append([pairs.iloc[i,0],pairs.iloc[i,0]])
#%% cointergation

# https://my.oschina.net/u/4586457/blog/4428026
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html?highlight=coint
# If the two series are almost perfectly collinear, then computing the test is numerically unstable. However,
# the two series will be cointegrated under the maintained assumption that they are integrated.
# In this case the t-statistic will be set to -inf and the pvalue to zero.
from statsmodels.tsa.stattools import coint

for pair_i in pairs_list:
    coint_t_value = coint(close.dropna().loc[:,pair_i[0]], close.dropna().loc[:,pair_i[1]])[0]
    print(coint(close.dropna().loc[:,pair_i[0]], close.dropna().loc[:,pair_i[1]]))
    if coint_t_value != '-inf':
        pairs_list.remove(pair_i)

#%% Hurst
# https://blog.csdn.net/weixin_30651273/article/details/96323309
def Hurst(data):
    n = 6
    data = pd.Series(data).pct_change()[1:]
    ARS = list()
    lag = list()
    for i in range(n):
        m = 2 ** i
        size = np.size(data) // m
        lag.append(size)
        panel = {}
        for j in range(m):
             panel[str(j)] = data[j*size:(j+1)*size].values

        panel = pd.DataFrame(panel)
        mean = panel.mean()
        Deviation = (panel - mean).cumsum()
        maxi = Deviation.max()
        mini = Deviation.min()
        sigma = panel.std()
        RS = maxi - mini
        RS = RS / sigma
        ARS.append(RS.mean())

    lag = np.log10(lag)
    ARS = np.log10(ARS)
    hurst_exponent = np.polyfit(lag, ARS, 1)
    hurst = hurst_exponent[0]

    return hurst

for pair_i in pairs_list:
    hurst = Hurst(close.dropna().loc[:,pair_i[0]]-close.dropna().loc[:,pair_i[1]])
    print(pair_i,'hurst=',hurst)
    if hurst < 0.5:
        pairs_list.remove(pair_i)

#%% half-life of mean-reversion
# https://blog.csdn.net/qq_26948675/article/details/115231586
import statsmodels.api as sm
def get_halflife(s):
    s_lag = s.shift(1)
    s_lag.iloc[0] = s_lag.iloc[1]
    s_ret = s - s_lag
    s_ret.iloc[0] = s_ret.iloc[1]
    s_lag2 = sm.add_constant(s_lag)
    model = sm.OLS(s_ret, s_lag2)
    res = model.fit()
    # print(res.summary())

    halflife = round(-np.log(2) / list(res.params)[1], 0)
    return halflife
#%%
for pair_i in pairs_list:
    halflife_0 = get_halflife(close.dropna().loc[:,pair_i[0]])
    halflife_1 = get_halflife(close.dropna().loc[:,pair_i[1]])

    print(pair_i,'halflife_0=',halflife_0,'halflife_1=',halflife_1)

    if halflife_0 > 252 or halflife_0 < 1 or halflife_1 > 252 or halflife_1 < 1:
        pairs_list.remove(pair_i)

#%% get coint coeff
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS
import statsmodels.api as sm

coint_coeff = {}
for pair_i in pairs_list:
    X = sm.add_constant(close.dropna().loc[:,pair_i[0]])
    model = sm.OLS(close.dropna().loc[:,pair_i[1]],X)
    coint_coeff[pair_i[0],pair_i[1]] = model.fit().params[1]
    print('coeff=',model.fit().params[1])


#%% mean-cross > 12
for pair_i in pairs_list:
    df = pd.concat([close.dropna().loc[:, pair_i[0]],close.dropna().loc[:,pair_i[1]]],axis=1)
    df['minus'] = df.apply(lambda x: x[0]-x[1] * coint_coeff[pair_i[0],pair_i[1]],axis=1)
    count_cross =  df['minus'].sum()
    print('count_cross=',count_cross)
    if count_cross < 12:
        pairs_list.remove(pair_i)

#%% trade
for pair_i in pairs_list:
    df = pd.concat([close.dropna().loc[:, pair_i[0]], close.dropna().loc[:, pair_i[1]]], axis=1)
    df['minus'] = df.apply(lambda x: x[0] - x[1] * coint_coeff[pair_i[0], pair_i[1]], axis=1)
    df['minus'][df['minus'] < 0] = 0
    df['nav'] = (df['minus']+1).cumprod()
    print(pair_i,'nav=',df['nav'].iloc[-1])



