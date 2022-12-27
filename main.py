#%%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataprep.eda import create_report
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import xgboost

#모듈 가져오기
from md_sql import pushToDB, pullFromDB
from md import dateSplitter, getGrade, lego, getLCplot

# import lightgbm as lgb

#%%
file = 'data/powercomp.csv'
password = '********'
dbName = 'electricityDB'
tableName = 'test'
ifexists = 'fail' #append(추가) / fail(있으면 에러) / replace(대체)

pushToDB(file, password, dbName, tableName, ifexists)

#%%

password = '********'
dbName = 'electricityDB'
tableName = 'powercomp'

df = pullFromDB(password, dbName, tableName)
print(f'powercomp shape: {df.shape}')
df.head()


# %%
##auto visualization
# powerReport = create_report(powercomp)
# powerReport.save('powercomp_summary.html')
# %%
#<Process>
##### 결측처리: 없음(완료)
##### 레이블링 및 더미 처리
##### 파생변수 생성
##### 스케일링(dayname 더미 변환)
# 컬럼 제거: datetime, year, humidity_grade_q
# 더미 처리: dayname, humidity_grade
# 스케일링: power_consumption 제외 나머지 전부
##### 데이터 분리
##### 모델 정의 / 학습 / 평가
##### 하이퍼 파라미터 튜닝
##### 최종 결과 산출
#%%
#Target1까지만 남기기
df = df.iloc[:,0:7]
df.head()
print(f'df shape: {df.shape}')

# %%
df.rename(columns={'Zone 1 Power Consumption' : 'power consumption'}, inplace=True)
df.columns = df.columns.str.lower()
df.head()

# %%
##레이블링 및 더미 처리
df['datetime'] = pd.to_datetime(df['datetime'])
df.info()
# %%
#datetime컬럼 확인
df['datetime']
# %%
df = df
dateCol = 'datetime'
df = dateSplitter(df, dateCol)
df.head()
# %%
df = df
gradeList = [1, 2, 3, 4, 5]
beGradeCol = 'humidity'
num = 5
getGrade(df, gradeList, beGradeCol, num)
df.head()
#%%
#구간화된 컬럼 확인
print(df['humidity_grade'].value_counts())
print(df['humidity_grade_q'].value_counts())

# %%
#월, 시, 요일에 따른 전력 소비량 시각화
plt.figure(figsize=(20,15))
plotList = ['month', 'hour', 'dayname']
for idx, col in enumerate(plotList):
    plt.subplot(3, 1, idx+1)
    sns.lineplot(data=df, x=col, y='power consumption', ci=None)
    plt.tight_layout
# %%
#상관관계 확인
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), cmap='coolwarm', mask=mask, annot=True, linewidth=0.5, vmax=1, vmin=-1)

#%%
plt.figure(figsize=(20,15))
df['power consumption'].plot()
plt.show()
# %%
fig = make_subplots(rows=1, cols=1, subplot_titles=('Power Consumption by day'))
fig.add_trace(go.Line(x=df['datetime'], y=df['power consumption'], name='open'), row=1, col=1)
# %%
#요일/시간별 전력 소비량 확인
##시간별 전력소비량 변화의 흐름은 비슷하나, 일요일의 경우 다른 요일에 비해 전력소비량이 낮음
##휴일인 일요일의 경우에만 다른 요일들에 비해 전력소비량이 낮은 것으로 보아,
## 가정용 전력소비량보다 공업/산업용 전력소비량이 전체 전력소비량 변동에 더 많은 영향을 끼칠 것이라고 예측
plt.figure(figsize=(20,15))
sns.pointplot(data=df, x=df['hour'], y='power consumption', hue=df['dayname'], ci=None)
# %%
#시간, 온도 boxplot
plt.subplot(1,2,1)
sns.boxplot(df['hour'])
plt.subplot(1,2,2)
sns.boxplot(df['temperature'])

# %%
# 컬럼 제거: datetime, year, humidity_grade_q
df = df.drop(['datetime', 'year', 'humidity_grade_q'], axis=1)
df.head()
#%%
# 더미 처리: dayname, humidity_grade
df = pd.get_dummies(df, columns=['dayname'])
df = pd.get_dummies(df, columns=['humidity_grade'])

#%%
# 스케일링: power_consumption(target) 제외 나머지 전부
mmscaler = MinMaxScaler()

cols = list(df)  #dataframe에 list를 씌우면 컬럼명들이 리스트로 반환
cols.remove("power consumption") #target은 스케일링에서 제외
df[cols] = mmscaler.fit_transform(df[cols])
# %%
df.head()
# %%
#학습을 위한 holdout(data split)
X = df.drop(['power consumption'], axis=1)
y = df['power consumption']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

#%%
lreg = LinearRegression()
rf = RandomForestRegressor()
xgb = xgboost.XGBRegressor()
vtr = VotingRegressor([
    ('linear', lreg),
    ('Random Forest', rf),
    ('Xgboost', xgb)
])

mse = mean_squared_error

algos = [lreg, rf, xgb, vtr]
for algo in algos:
    lego(algo, X_train, X_test, y_train, y_test)

#%%
algoList = [lreg, rf, xgb, vtr]
row = 2
col = 2
getLCplot(algoList, X_train, y_train, row, col)
