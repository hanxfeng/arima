import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import itertools
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
import chardet
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']


class Error(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

#仅支持10%，5%，1%三种输入
#进行kpss和adf两种检验，均通过则认为平稳，有一种不通过则认为不平稳
def data_adf(data,c='5%',ft=False):

    #报错
    if c != '1%' and c != '5%' and c != '10%':
        raise Error('c仅支持10%，5%，1%三种输入')
    if ft != True and ft != False:
        raise Error('ft仅支持输入True或False')

    warnings.filterwarnings('ignore', category=UserWarning,
                            message='The test statistic is outside of the range of p-values')#用于忽略警告

    result1 = adfuller(data)  #adf检验
    result2 = kpss(data)     #kpss检验

    #adf检验下的差分次数
    re = result1[4][c]
    i=0
    while result1[0]>re:
        i = i + 1
        data1=data.diff(i)
        data1=data1.fillna(0)
        result1=adfuller(data1)

    #kpss检验下的差分次数
    res=result2[3][c]
    j=0
    while result2[0]>res:
        j=j+1
        data2=data.diff(j)
        data2=data2.fillna(0)
        result2=kpss(data2)

    #判断数据需要的差分次数，以kpss和adf差分次数较高的为最后结果
    if i>=j:
        result=result2
        name='adf'
    else:
        i=j
        result=result1
        name='kpaa'

    #画图
    if ft:
        plt.figure()
        plt.plot(range(len(data)),data)
        plt.show()

    #输出结果
    if i==0:
        print('数据平稳，无需差分')
        print(f'{name}检验的统计量与临界值:')
        print(result)
        print('----------------------------')
        return 0
    else:
        print(f'数据不平稳，进行{i}次差分后平稳')
        print(f'{name}检验的统计量与临界值:')
        print(result)
        print('----------------------------')
        return i

#白噪声检验
def lb(data,sl=0.05):

    #进行白噪声检验
    lb = sm.stats.diagnostic.acorr_ljungbox(data, return_df=False)

    #输出白噪声检验结果
    if all(lb['lb_pvalue'] > sl):
        print("数据可能是白噪声。")
    else:
        print("数据不是白噪声。")

    #输出白噪声检验统计量与p值
    print('白噪声检验的统计量与p值')
    print(lb)
    print('----------------------------')

#name有AIC，BIC，ALL三种输入，当name=ALL时返回的p，q为BIC下的p，q
#获取数据最佳p，q
def train_pq(data,d,plt_=False,name='BIC'):

    #报错
    if plt_!=False and plt_!=True:
        raise Error('plt_仅支持输入True或False')
    if name!='AIC' and name!='BIC' and name!='ALL':
        raise Error('name仅支持输入AIC，BIC，ALL')

    #忽略警告
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='Maximum Likelihood optimization failed to converge')
    warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found.')
    warnings.filterwarnings('ignore', message='Non-stationary starting autoregressive parameters')

    train_results = sm.tsa.arma_order_select_ic(data, ic=['aic', 'bic'], trend='c', max_ar=9, max_ma=9)

    #不画图可以减少程序运行时间
    if plt_==True:
        p_min = 0
        q_min = 0
        p_max = 9
        q_max = 9
        d = d
        res_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                               columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
        res_aic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                               columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
        for p, q in itertools.product(range(p_min, p_max + 1),
                                      range(q_min, q_max + 1)):
            if p == 0 and q == 0:
                res_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
                continue
            try:
                model = ARIMA(data, order=(p, d, q))
                results = model.fit()
                res_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
                res_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
            except:
                continue
        results_bic = res_bic[res_bic.columns].astype(float)  # 转化为浮点数
        results_aic = res_bic[res_aic.columns].astype(float)
        if name=='AIC':
            fig, ax = plt.subplots(figsize=(10, 8))
            ax = sns.heatmap(results_aic,
                             mask=results_aic.isnull(),
                             ax=ax,
                             annot=True,
                             fmt='.2f',
                             )
            ax.set_title('AIC')
            plt.show()
        elif name=='BIC':
            fig, ax = plt.subplots(figsize=(10, 8))
            ax = sns.heatmap(results_bic,
                             mask=results_bic.isnull(),
                             ax=ax,
                             annot=True,
                             fmt='.2f',
                             )
            ax.set_title('BIC')
            plt.show()
        elif name=='ALL':
            fig, ax = plt.subplots(figsize=(10, 8))
            ax = sns.heatmap(results_aic,
                             mask=results_aic.isnull(),
                             ax=ax,
                             annot=True,
                             fmt='.2f',
                             )
            ax.set_title('AIC')
            plt.show()

            fig, ax = plt.subplots(figsize=(10, 8))
            ax = sns.heatmap(results_bic,
                             mask=results_bic.isnull(),
                             ax=ax,
                             annot=True,
                             fmt='.2f',
                             )
            ax.set_title('BIC')
            plt.show()

        # 根据选择的方式输出结果
        if name == 'BIC':
            print(f'BIC:{train_results.bic_min_order}')
            return list(train_results.bic_min_order)[0], list(train_results.bic_min_order)[1]
        elif name == 'AIC':
            print(f'AIC:{train_results.aic_min_order}')
            return list(train_results.aic_min_order)[0], list(train_results.aic_min_order)[1]
        elif name == 'ALL':
            print(f'BIC:{train_results.bic_min_order}')
            print(f'AIC:{train_results.aic_min_order}')
            return list(train_results.bic_min_order)[0], list(train_results.bic_min_order)[1]
        else:
            print('输入不正确')

        return list(train_results.bic_min_order)[0],list(train_results.bic_min_order)[1]

#进行预测
def train_(data,steps,p,d,q,pl=False,writer=False):

    #报错
    if pl!=True and pl!=False:
        raise Error('pl仅支持输入True或False')
    if writer!=True and writer!=False:
        raise Error('writer仅支持输入True或False')

    #忽略警告
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='Maximum Likelihood optimization failed to converge')
    warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found.')
    warnings.filterwarnings('ignore', message='Non-stationary starting autoregressive parameters')

    #划分训练集与测试集
    train,test=train_test_split(data,train_size=0.8,test_size=0.2,shuffle=False)
    len_train=int(len(train))
    len_test=int(len(test))

    #拟合模型
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    resid=model_fit.resid#获取残差

    #正态检验
    w,p_value=shapiro(resid)
    print('-----------正态检验-------------')
    if p<0.05:
        print('残差不是正态分布，模型拟合不合理')
    else:
        print('残差符合正态分布，模型拟合合理')
    print(f'统计量:{w}')
    print(f'p值:{p}')

    #自相关性检验
    print('----------自相关检验-------------')
    dw=sm.stats.durbin_watson(resid.values)
    if dw>=1 and dw<=3:
        print('残差不存在自相关性')
    else:
        print('残差存在自相关性')

    #白噪声检验
    lb = sm.stats.diagnostic.acorr_ljungbox(resid, return_df=False)

    print('----------白噪声检验-------------')
    if all(lb['lb_pvalue'] > 0.05):
        print("数据可能是白噪声。")
    else:
        print("数据不是白噪声。")
    print(lb)

    #通过测试集获取模型准确率
    pre=model_fit.predict(start=len_train,end=len_train+len_test-1)
    MAE=np.linalg.norm(pre - test, ord=1) / len(pre)
    RMSE=np.sqrt(((pre - test) ** 2).mean())

    print('------------准确率-------------')
    print(f'MAE:{MAE}')
    print(f'RMSE:{RMSE}')
    print('----------------------------')

    #进行预测
    predict=model_fit.predict(start=len(data),end=len(data)+steps-1)

    print('------------预测结果------------')
    print(predict)

    #画图
    if pl:
        plt.figure()
        plt.plot(range(len(data)),data,label='实际')
        plt.plot(range(len_train+1,len(data)+1),pre,label='预测')
        plt.title('测试集')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(range(len(predict)),predict)
        plt.title('预测结果')
        plt.show()

        #保存结果
        if writer!=False:
            p=pd.DataFrame({'预测值':predict})
            p.to_excel(writer)

    return predict

#获取文件编码方式
def bianma(path):

    with open(path, 'rb') as f:
        rawdata = f.read(1310)
        result = chardet.detect(rawdata)

    return result['encoding']


