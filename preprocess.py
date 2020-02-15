import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date


train = pd.read_csv("train.csv")
wtr = pd.read_csv("weather.csv")
key = pd.read_csv("key.csv")

#1st file
train['log1p'] = np.log(train['units'] + 1)

group = train.groupby(["store_nbr", "item_nbr"])['log1p'].mean()
group = group[group > 0.0]
store_nbrs = group.index.get_level_values(0)
item_nbrs = group.index.get_level_values(1)

store_item_nbrs = sorted(zip(store_nbrs, item_nbrs), key = lambda t: t[1] * 10000 + t[0] )

with open('train1.csv', 'wb') as f: 
	f.write("store_nbr,item_nbr\n")
	for sno, ino in store_item_nbrs:
        	f.write("{},{}\n".format(sno, ino))

train['date2j'] = np.ones(len(train['date']))
print ("start")
train['date2j'] = (pd.to_datetime(train.date) - pd.to_datetime("2012-01-01")).dt.days
train = train[train.date != '2013-12-25']
train.to_csv('baseline.csv')

print ("Baseline")

#2nd file
#For holidays
f = open("holidays.txt")
lines = f.readlines()
lines = [line.split(" ")[:3] for line in lines]
lines = ["{} {} {}".format(line[0], line[1], line[2]) for line in lines]
lines = pd.to_datetime(lines)
holidays = pd.DataFrame({"date2":lines})
#print (holidays)

#For holidays names
f = open("holiday_names.txt")
lines = f.readlines()
lines = [line.strip().split(" ")[:4] for line in lines]
lines_dt = ["{} {} {}".format(line[0], line[1], line[2]) for line in lines]
lines_dt = pd.to_datetime(lines_dt)
lines_hol = [line[3] for line in lines]
holiday_names = pd.DataFrame({"date2":lines_dt, "holiday_name":lines_hol})
#print(holidays_names)

print("Holidays done")

#Preprocessing the weather file
wtr['date2'] = pd.to_datetime(wtr.date)
wtr["preciptotal_flag"] = np.where(wtr["preciptotal"] =='M', 0.0, wtr["preciptotal"])
wtr["preciptotal_flag"] = np.where(wtr["preciptotal"] == 'T', 0.0, wtr["preciptotal"])
wtr["preciptotal_flag"] = np.where(wtr["preciptotal"] > 0.2, 1.0, 0.0)
wtr["depart2"] = np.where(wtr.depart=='M', np.nan, wtr.depart)
wtr["depart2"] = np.where(wtr.depart=='T', 0.00, wtr.depart)
wtr["depart_flag"] = 0.0
wtr["depart_flag"] = np.where(wtr["depart2"] < -8.0, -1, wtr["depart_flag"])
wtr["depart_flag"] = np.where(wtr["depart2"] > 8.0 ,  1, wtr["depart_flag"])

print ("Weather done")

#store_item_nbrs = pd.read_csv('train1.csv')
valid_store_items = set(store_item_nbrs)
mask_train = [(sno_ino in valid_store_items) for sno_ino in zip(train['store_nbr'], train['item_nbr']) ]
train = train[mask_train].copy()

#Preprocessing the train file
train['date2'] = pd.to_datetime(train['date'])
train = pd.merge(train, key, on='store_nbr')
train = pd.merge(train, wtr[["date2", "station_nbr", "preciptotal_flag", "depart_flag"]], 
                      on=["date2", "station_nbr"])

#Checking for weekday and holiday
train['weekday'] = train.date2.dt.weekday
train['is_weekend'] = train.date2.dt.weekday.isin([5,6])
train['is_holiday'] = train.date2.isin(holidays.date2)
train['is_holiday_weekday'] = train.is_holiday & (train.is_weekend == False)
train['is_holiday_weekend'] = train.is_holiday &  train.is_weekend

#Making binary class in form of 0 and 1
train.is_weekend = np.where(train.is_weekend, 1, 0)
train.is_holiday = np.where(train.is_holiday, 1, 0)
train.is_holiday_weekday = np.where(train.is_holiday_weekday, 1, 0)
train.is_holiday_weekend = np.where(train.is_holiday_weekend, 1, 0)

# day, month, year
train['day'] = train.date2.dt.day
train['month'] = train.date2.dt.month
train['year'] = train.date2.dt.year

# around BlackFriday
train = pd.merge(train, holiday_names, on='date2', how = 'left')
train.loc[train.holiday_name.isnull(), "holiday_name"] = ""

near_BlackFriday = ["BlackFridayM3", "BlackFridayM2", "ThanksgivingDay", "BlackFriday",
                  "BlackFriday1", "BlackFriday2", "BlackFriday3"]
train["around_BlackFriday"] = np.where(train.holiday_name.isin(near_BlackFriday), 
                                train.holiday_name, "No")
train.to_csv('processed_train.csv')



#Preprocessing test file
test = pd.read_csv("test.csv")

test['id'] = range(len(test))
test['newid'] = (test['store_nbr'].map(str) + "_" + test['item_nbr'].map(str) + "_" + test['date'].map(str))


test['date2j'] = (pd.to_datetime(test.date) - pd.to_datetime("2012-01-01")).dt.days

#store_item_nbrs = pd.read_csv('train1.csv')
mask_test = [(sno_ino in valid_store_items) for sno_ino in zip(test['store_nbr'], test['item_nbr']) ]
test = test[mask_test].copy()

#Preprocessing the test file
test['date2'] = pd.to_datetime(test['date'])
test = pd.merge(test, key, on='store_nbr')
test = pd.merge(test, wtr[["date2", "station_nbr", "preciptotal_flag", "depart_flag"]], 
                      on=["date2", "station_nbr"])

#Checking for weekday and holiday
test['weekday'] = test.date2.dt.weekday
test['is_weekend'] = test.date2.dt.weekday.isin([5,6])
test['is_holiday'] = test.date2.isin(holidays.date2)
test['is_holiday_weekday'] = test.is_holiday & (test.is_weekend == False)
test['is_holiday_weekend'] = test.is_holiday &  test.is_weekend

#Making binary class in form of 0 and 1
test.is_weekend = np.where(test.is_weekend, 1, 0)
test.is_holiday = np.where(test.is_holiday, 1, 0)
test.is_holiday_weekday = np.where(test.is_holiday_weekday, 1, 0)
test.is_holiday_weekend = np.where(test.is_holiday_weekend, 1, 0)

# day, month, year
test['day'] = test.date2.dt.day
test['month'] = test.date2.dt.month
test['year'] = test.date2.dt.year

# around BlackFriday
test = pd.merge(test, holiday_names, on='date2', how = 'left')
test.loc[test.holiday_name.isnull(), "holiday_name"] = ""

test["around_BlackFriday"] = np.where(test.holiday_name.isin(near_BlackFriday), 
                                test.holiday_name, "No")
test.to_csv('processed_test.csv')


