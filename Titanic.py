
# coding: utf-8

# # [kaggleのタイタニック]
#
# 以下、今回の解析で重要なものについて。
# + Survived ... 1 -> 生存者(Alive), 0 ->死者(Dead)
# + Pclass ... 1,2,3の順に高級クラスの客室
# + Sex,Age ... それぞれ性別と年齢
# + SibSp ... 兄弟および配偶者の数
# + Parch ... 親もしくは子供の数
# + ticket... チケット番号
# + Fare ... 運賃
# + Embarked ... 乗船した港（３つ）
#
# ## Xgboostをやってみる

# In[2]:


# 必要なものをインポートする
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

from collections import Counter

import re


# jupyter-notebookに描画する
#get_ipython().magic('matplotlib inline')


# ### 入力データを読み込む

# In[3]:


titanic_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# ### 入力データの情報を出力

# In[4]:


titanic_df.info()


# In[5]:


titanic_df.head()


# ### 文字情報は解析に使用しにくいため、置換する

# In[6]:


titanic_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Sexにを値に置換
titanic_df['Sex'] =titanic_df['Sex'].replace('male',1)
titanic_df['Sex'] =titanic_df['Sex'].replace('female',2)

test_df['Sex'] =test_df['Sex'].replace('male',1)
test_df['Sex'] =test_df['Sex'].replace('female',2)


# Embarkedにを値に置換
Embarked_map_key = {'S':1,'C':2,'Q':3}
combine=[titanic_df]
for titanic_df in combine:
    titanic_df['Embarked']=titanic_df['Embarked'].map(Embarked_map_key)
    titanic_df['Embarked']=titanic_df['Embarked'].fillna(0)

combine=[test_df]
for test_df in combine:
    test_df['Embarked']=test_df['Embarked'].map(Embarked_map_key)
    test_df['Embarked']=test_df['Embarked'].fillna(0)


# Nameに入っている　「○○.」を Salutation列を作って入れる
tmp=[]
for tmps in titanic_df['Name'].str.split(', '):
    tmp.append(tmps[1])

Salutation =[]
for Salutations in tmp:
    Salutation.append(Salutations.split('.')[0])

titanic_df['Salutation']=Salutation

combine=[titanic_df]

# Salutationから抜き出した内容をMr Miss Mrsなどに置換
combine=[titanic_df]

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
for titanic_df in combine:
        titanic_df['Salutation'] = titanic_df['Salutation'].map(Salutation_mapping)
        titanic_df['Salutation'] = titanic_df['Salutation'].fillna(0)


tmp=[]
for tmps in test_df['Name'].str.split(', '):
    tmp.append(tmps[1])

Salutation =[]
for Salutations in tmp:
    Salutation.append(Salutations.split('.')[0])

test_df['Salutation']=Salutation

# Salutationから抜き出した内容をMr Miss Mrsなどに置換
combine=[test_df]

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
for test_df in combine:
        test_df['Salutation'] = test_df['Salutation'].map(Salutation_mapping)
        test_df['Salutation'] = test_df['Salutation'].fillna(0)


# In[7]:


# NULL値を補正
titanic_df['Age'].fillna(titanic_df.Age.mean(), inplace=True)
test_df['Age'].fillna(test_df.Age.mean(),inplace=True)

test_df['Fare'].fillna(titanic_df.Pclass.mean(),inplace=True)
test_df['Fare'].fillna(0,inplace=True)


# In[8]:


combine=[titanic_df]
titanic_df["FamilySize"] = titanic_df["SibSp"] + titanic_df["Parch"] + 1
for titanic_df in combine:
    titanic_df['IsAlone'] = 0
    titanic_df.loc[titanic_df['FamilySize'] == 1, 'IsAlone'] = 1

combine=[test_df]
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
for test_df in combine:
    test_df['IsAlone'] = 0
    test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1

test_data = test_df.values


# In[9]:


key =['Pclass','Sex','Age','SibSp','Parch','Fare','Salutation','IsAlone']
xs = titanic_df[key].values # Pclass以降の変数
y  = titanic_df['Survived']  # 正解データ

xs_test=test_df[key].values


# In[11]:


from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier()
random_forest.fit(xs, y)
Y_pred = random_forest.predict(xs_test)

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_df['PassengerId'].values.astype(int), Y_pred.astype(int)):
        writer.writerow([pid, survived])


# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV

parameters = {
        'n_estimators'      : [10,25,50,75,100],
        'random_state'      : [0],
        'n_jobs'            : [4],
        'min_samples_split' : [5,10, 15, 20,25, 30],
        'max_depth'         : [5, 10, 15,20,25,30]
}

clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(xs, y)

print(clf.best_estimator_)
