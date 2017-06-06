import pandas as pd
import numpy as np
import zipfile
import os
import glob
from math import log
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import time
import scipy
from sklearn.externals import joblib # create pickles for models, avoid retrains
import operator
from sklearn import linear_model
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix # classification evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score # regression evaluation
import pylab 
import xgboost as xgb
from gensim.utils import tokenize
import gensim
import pyemd
# setup visual output configs
sns.set_style('darkgrid')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.options.display.max_colwidth = 200
pd.set_option('display.width', 1000)
#import warnings
#warnings.filterwarnings('ignore')
import time

start = time.time()

train = pd.read_pickle('./data/train_processed.pkl')
test = pd.read_pickle('./data/test_processed.pkl')

test['q1_w2v'] = test['q1_w2v'].apply(pd.Series)
test['q2_w2v'] = test['q2_w2v'].apply(pd.Series)
test.to_pickle("./data/test_processed1.pkl")
del test

train['q1_w2v'] = train['q1_w2v'].apply(pd.Series)
train['q2_w2v'] = train['q2_w2v'].apply(pd.Series)
train.to_pickle("./data/train_processed.pkl")
print(str(time.time() - start) + "s")
