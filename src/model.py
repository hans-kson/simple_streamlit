import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df=pd.read_csv('../data/data.csv')

y=df.Species.copy()
X=df.copy()
X.drop(['Species'], axis=1, inplace=True)
X = X.replace(["Brown", "Blue"], [1, 0])

clf = LogisticRegression() 
clf.fit(X,y)

joblib.dump(clf,'../model/clf.pkl')