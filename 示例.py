import pandas as pd
from 函数 import data_adf,lb,train_pq,train_,bianma
import statsmodels.api as sm
a=bianma('1.csv')
data=pd.read_csv('1.csv',encoding=a)
aa='货量'
data=data['货量']

d=data_adf(data)
lb(data)
p,q=train_pq(data,d)
a=train_(data,10,p,d,q)

