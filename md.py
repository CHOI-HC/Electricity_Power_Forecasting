#%%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px

from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import xgboost

#%%
#연도, 월, 일, 시, 요일 컬럼 생성
def dateSplitter(df, dateCol):
    df['year'] = df[dateCol].dt.year
    df['month'] = df[dateCol].dt.month
    df['day'] = df[dateCol].dt.day
    df['hour'] = df[dateCol].dt.hour
    df['dayname'] = df[dateCol].dt.day_name()
    return df

#구간화(cut/qcut)
def getGrade(df, gradeList, beGradeCol, num):
    labels = gradeList
    df[f'{beGradeCol}_grade'] = pd.cut(df[beGradeCol], num, labels=labels)
    df[f'{beGradeCol}_grade_q'] = pd.qcut(df[beGradeCol], num, labels=labels)
    return df

#학습 및 rmse, r2 확인
def lego(algo, X_train, X_test, y_train, y_test):
    algo.fit(X_train, y_train)
    train_pred = algo.predict(X_train)
    test_pred = algo.predict(X_test)
    train_rmse = np.sqrt(mse(y_train, train_pred))
    test_rmse = np.sqrt(mse(y_test, test_pred))
    train_r2_score = r2_score(y_train, train_pred)
    test_r2_score = r2_score(y_test, test_pred)
    print(f"\n{algo.__class__.__name__} train rmse score: {train_rmse}")
    print(f"{algo.__class__.__name__} train r2 score: {train_r2_score}")
    print(f"{algo.__class__.__name__} test rmse score: {test_rmse}")
    print(f"{algo.__class__.__name__} test r2 score: {test_r2_score}")

#Learning Curve 확인
def getLCplot(algoList, X_train, y_train, row, col):
    plt.figure(figsize=(20,15))
    for idx, algo in enumerate(algoList):
        trainSizes, trainScores, testScores = learning_curve(algo, X_train, y_train, train_sizes=np.linspace(.1,1,5), cv=3, scoring="neg_mean_squared_error", n_jobs=1)
        trainScoresMean = np.mean(np.sqrt(-trainScores), axis=1)
        testScoresMean = np.mean(np.sqrt(-testScores), axis=1)
        plt.subplot(row, col,idx+1)
        plt.plot(trainSizes, trainScoresMean, "-o", label="train rmse")
        plt.plot(trainSizes, testScoresMean, "-o", label="test rmse")
        plt.legend(loc='best')
        plt.title(algo.__class__.__name__)
    plt.tight_layout()