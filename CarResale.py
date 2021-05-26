import pandas as pd
data = pd.read_csv(r'C:\Users\Parth Tiwari\Downloads\Car_Price_deployment-main\Car_Price_deployment-main\car_data.csv')
data = data[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
data['Current_Year']=2021
data['Number_of_Years_Used']=data['Current_Year']-data['Year']
data.drop(['Year'] , axis=1 , inplace=True)
data.drop(['Current_Year'] , axis=1 , inplace=True)
final_dataset = pd.get_dummies(data,drop_first=True)
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200 ,num = 12)]

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200 ,num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30 ,num = 6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
               }
rf=RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)
rf_random.fit(X_train,y_train) 

predictions = rf_random.predict(X_test)

import pickle
file = open('random_forest_regression_model.pkl','wb')
pickle.dump(rf_random,file)