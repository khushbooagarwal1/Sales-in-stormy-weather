import pandas as pd
import numpy as np
from sklearn.svm import SVR

#Read train file
train = pd.read_csv('processed_train.csv', low_memory=False)

train['sno_ino'] = (train['store_nbr']*1000 + train['item_nbr'])/1000
train['near_blackfriday']=np.where(train.around_BlackFriday=="No",0,1)


train_data_input = train[['sno_ino','date2j','depart_flag','preciptotal_flag','weekday','is_weekend','is_holiday','is_holiday_weekend','is_holiday_weekday','day','month','year','near_blackfriday']]

train_data_output = train['log1p']

#Read test file
test = pd.read_csv('processed_test.csv', low_memory=False)

test['sno_ino'] = (test['store_nbr']*1000 + test['item_nbr'])/1000
test['near_blackfriday']=np.where(test.around_BlackFriday=="No",0,1)


test_data_input = test[['sno_ino','date2j','depart_flag','preciptotal_flag','weekday','is_weekend','is_holiday','is_holiday_weekend','is_holiday_weekday','day','month','year','near_blackfriday']]

print "start"
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#test_data_output_log = svr_rbf.fit(train_data_input, train_data_output).predict(test_data_input)
test_data_output_log = svr_lin.fit(train_data_input, train_data_output).predict(test_data_input)
#y_poly = svr_poly.fit(X, y).predict(X)

test_data_output = (np.exp(test_data_output_log)).astype(int)
#print test_data_output
print "prediction done"
test['units']=test_data_output
#print test['units']

test['newid'] = (test['store_nbr'].map(str) + "_" + test['item_nbr'].map(str) + "_" + test['date'].map(str))
submit = pd.read_csv("output.csv")

submit = pd.merge(submit, test[['newid','units']], on='newid', how = 'left')
submit['units'] = np.where(submit['units_y']>0,submit['units_y'],0)
submit = submit.drop('units_y', axis=1)
submit = submit.drop('units_x', axis=1)
submit.to_csv('finaloutput.csv')

