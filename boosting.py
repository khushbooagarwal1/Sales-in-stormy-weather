import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

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


#Gradient Boosting
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=16, random_state=0, loss='ls').fit(train_data_input, train_data_output)
test_data_output_log = model.predict(test_data_input)
test_data_output = (np.exp(test_data_output_log)).astype(int)
#print test_data_output

test['units']=test_data_output
print test['units']

test['newid'] = (test['store_nbr'].map(str) + "_" + test['item_nbr'].map(str) + "_" + test['date'].map(str))
submit = pd.read_csv("output.csv")

submit = pd.merge(submit, test[['newid','units']], on='newid', how = 'left')
submit['units'] = np.where(submit['units_y']>0,submit['units_y'],0)
submit = submit.drop('units_y', axis=1)
submit = submit.drop('units_x', axis=1)
submit.to_csv('finaloutput.csv')



