
# Credit Card Fraud Detection

# Tested different Machine Learning models on past transactions of valid and fraud transactions
# Trained the data using K-Means, Support Vector Classifiers
# Used Multivariate Normal to get a precision of 91%

# https://www.kaggle.com/gauravduttakiit/creditcard-fraud-detection-by-logistic-regression



import pandas as pd
import numpy as np

df = pd.read_csv('creditcard.csv')
df # display dataframe


df.describe() # describe() method is used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame


df.isnull().sum() # returns the number of missing values in the data set # dtype: int64

# df = df.drop([49609],axis=0)  # drop() function is used to drop specified labels from rows; axis=0 for removing row; [rows to remove]

df.drop(df.index[49601:49606], axis=0, inplace=True) # remove rows from r1 to r2

df # display dataframe

df.isnull().sum()

fraud = df[df['Class'] == 1]

valid = df[df['Class'] == 0]

df['Class'].value_counts()
'''
0.0    49453
1.0      148
Name: Class, dtype: int64
'''


outlierFraction = len(fraud)/float(len(valid)) 



print(outlierFraction) 
print('Fraud Cases: {}'.format(len(df[df['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(df[df['Class'] == 0])))
'''
0.002992740581966716
Fraud Cases: 148
Valid Transactions: 49453
'''




fraud.Amount.describe()
'''
count     148.000000
mean      100.170676
std       233.347471
min         0.000000
25%         1.000000
50%         9.560000
75%        99.990000
max      1809.680000
Name: Amount, dtype: float64
'''

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

corrmat = df.corr() 


fig = plt.figure(figsize = (12, 9)) 
sns.heatmap(corrmat, vmax = .8, square = True) 
plt.show()

X = df.drop(['Class'], axis = 1)
Y = df["Class"]
print(X.shape) 
print(Y.shape) 
###
# (284807, 30)
# (284807,)
###

X

Y

from sklearn.model_selection import train_test_split # split the data into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

from sklearn.cluster import KMeans

sse = []
k_rng = range(1,10)


for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X_train)
    sse.append(km.inertia_)


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

km=KMeans(n_clusters=4)
km.fit(X)
km.labels_
# array([2, 2, 2, ..., 1, 1, 1], dtype=int32)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams


rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]



data1= df.sample(frac = 0.1,random_state=1)
state = np.random.RandomState(42)
Fraud = data1[data1['Class']==1]
Valid = data1[data1['Class']==0]
outlier_fraction = len(Fraud)/float(len(Valid))


classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1)
   
}



from sklearn.preprocessing import MinMaxScaler

classes = df['Class']
df.drop(['Time', 'Class', 'Amount'], axis=1, inplace=True)

df

cols = df.columns.difference(['Class'])
cols
'''
Index(['V1', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
       'V19', 'V2', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
       'V28', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9'],
      dtype='object')
'''

MMscaller = MinMaxScaler()

MMscaller
# MinMaxScaler()

df = MMscaller.fit_transform(df)

df
'''
array([[0.94311432, 0.799158  , 0.95776926, ..., 0.27245327, 0.44160937,
        0.2207915 ],
       [0.98683115, 0.80288665, 0.89383469, ..., 0.33428694, 0.434375  ,
        0.22161462],
       [0.94313922, 0.78521531, 0.93718124, ..., 0.28227144, 0.43202161,
        0.21990117],
       ...,
       [0.95024791, 0.80537084, 0.93206346, ..., 0.45320942, 0.4444533 ,
        0.22199162],
       [0.88872612, 0.81957875, 0.79872605, ..., 0.24699824, 0.46148019,
        0.21820065],
       [0.98717703, 0.79705172, 0.87215104, ..., 0.2560979 , 0.4377923 ,
        0.22172848]])
'''


df = pd.DataFrame(data=df, columns=cols)

df = pd.concat([df, classes], axis=1)

df


def train_validation_splits(df):
    # Fraud Transactions
    fraud = df[df['Class'] == 1]
    # Normal Transactions
    normal = df[df['Class'] == 0]
    print('normal:', normal.shape[0])
    print('fraud:', fraud.shape[0])
    normal_test_start = int(normal.shape[0] * .2)
    fraud_test_start = int(fraud.shape[0] * .5)
    normal_train_start = normal_test_start * 2
    val_normal = normal[:normal_test_start]
    val_fraud = fraud[:fraud_test_start]
    validation_set = pd.concat([val_normal, val_fraud], axis=0)
    test_normal = normal[normal_test_start:normal_train_start]
    test_fraud = fraud[fraud_test_start:fraud.shape[0]]
    test_set = pd.concat([test_normal, test_fraud], axis=0)
    Xval = validation_set.iloc[:, :-1]
    Yval = validation_set.iloc[:, -1]
    Xtest = test_set.iloc[:, :-1]
    Ytest = test_set.iloc[:, -1]
    train_set = normal[normal_train_start:normal.shape[0]]
    Xtrain = train_set.iloc[:, :-1]
    return Xtrain.to_numpy(), Xtest.to_numpy(), Xval.to_numpy(), Ytest.to_numpy(), Yval.to_numpy()


def estimate_gaussian_params(X):
    '''
    Calculates the mean and the covariance for each feature.
    Arguments:
    X: dataset
    '''
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)
    return mu, sigma
    
    
    
from scipy.stats import multivariate_normal
(Xtrain, Xtest, Xval, Ytest, Yval) = train_validation_splits(df)
(mu, sigma) = estimate_gaussian_params(Xtrain)
# calculate gaussian pdf
p = multivariate_normal.pdf(Xtrain, mu, sigma)
pval = multivariate_normal.pdf(Xval, mu, sigma)
ptest = multivariate_normal.pdf(Xtest, mu, sigma)
'''
normal: 49453
fraud: 148
'''




def metrics(y, predictions):
    fp = np.sum(np.all([predictions == 1, y == 0], axis=0))
    tp = np.sum(np.all([predictions == 1, y == 1], axis=0))
    fn = np.sum(np.all([predictions == 0, y == 1], axis=0))
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, F1
    
    
    
def selectThreshold(yval, pval):
    e_values = pval
    bestF1 = 0
    bestEpsilon = 0
    for epsilon in e_values:
        predictions = pval < epsilon
        (precision, recall, F1) = metrics(yval, predictions)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1



(epsilon, F1) = selectThreshold(Yval, pval)

print("Best epsilon found:", epsilon)
print("Best F1 on cross validation set:", F1)
'''
Best epsilon found: 1.1259014736511131e-83
Best F1 on cross validation set: 0.023668639053254437
'''

(test_precision, test_recall, test_F1) = metrics(Ytest, ptest < epsilon)

print("Outliers found:", np.sum(ptest < epsilon))
print("Test set Precision:", test_precision)
print("Test set Recall:", test_recall)
print("Test set F1 score:", test_F1)
'''
Outliers found: 4309
Test set Precision: 0.015084706428405663
Test set Recall: 0.8783783783783784
Test set F1 score: 0.029660050193931097
'''




