import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.stats.moments import ewma
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression


def get_year_month(year_month):
    ''' splits the year_month column to year and month and returns them in int format'''
    years = []
    months = []
    for y_m in year_month:
        y,m = y_m.split('/')
        years.append(int(y))
        months.append(int(m))
    return years,months


def get_zero_repairs(st_year,end_year):
    import pandas as pd
    import itertools
    zero_df = []
    years = list(range(st_year,end_year +1))
    months = list(range(1,13))
    zero_df = [(y,m,0) for y,m in itertools.product(years,months)]
    zero_df = pd.DataFrame(zero_df)
    zero_df.columns = ['year','month','number_repair']
    return zero_df



repair_train = pd.read_csv('../data/RepairTrain.csv')
repair_train['year_sale'],repair_train['month_sale'] = get_year_month(repair_train['year/month(sale)'])
repair_train['year_repair'],repair_train['month_repair'] = get_year_month(repair_train['year/month(repair)'])
repair_min_year = repair_train['year_repair'].min()
repair_max_year = repair_train['year_repair'].max()

cols_requ = ['module_category','component_category','year_repair','month_repair','number_repair']
cols_groupby = ['module_category','component_category','year_repair','month_repair']
repair_train_summ = repair_train[cols_requ].groupby(cols_groupby).sum()

def get_repair_complete(module,component):
    zero_repair = get_zero_repairs(repair_min_year,repair_max_year)
    repair_module_component = repair_train[np.logical_and(
                                repair_train.module_category == module,
                                repair_train.component_category == component)]

    cols_req = ['year_repair','month_repair','number_repair']
    cols_groupby = ['year_repair','month_repair']
    cols_req_final = ['year','month','number_repair']
    repair_train_summ = repair_module_component[cols_req].groupby(cols_groupby,as_index=False).sum()
    repair_train_summ.columns = ['year','month','number_repair']
    repair_merged = pd.merge(zero_repair,repair_train_summ,how='left',on=['year','month'])
    repair_merged['number_repair'] = repair_merged['number_repair_x'] + repair_merged['number_repair_y']

    return repair_merged[cols_req_final]


pred_period = 19

def get_prediction(first_prediction, k, prediction_count):
    result = [first_prediction]

    for i in range(1, prediction_count):
        result.append(result[-1] * k)

    return result

def num_decreases(y):
    decreases = 0
    for i in range(1, y):
        if y.iloc[i] < y.iloc[i - 1]:
            decreases += 1

    return decreases


output_target = pd.read_csv('../data/Output_TargetID_Mapping.csv')
submission = pd.read_csv('../data/SampleSubmission.csv')


for i in range(0,output_target.shape[0],pred_period):
    module = output_target['module_category'][i]
    category = output_target['component_category'][i]
    X = get_repair_complete(module,category).fillna(0).query('year >= 2009 & month >= 5')
    years = X.year.apply(lambda y: (y-2009)*12)
    months = X.month.apply(lambda m: m - 5)
    x = years.astype(int).combine(months, func=lambda x, y: x + y)[:, np.newaxis]
    y = X.number_repair

    lr = LinearRegression()
    lr.fit(x, y.apply(lambda y: np.log(y + 1)))

    k = lr.coef_[0] if lr.coef_[0] <= 0 else np.log(0.91)
    k = np.exp(k)

    decreases = num_decreases(y)
    first_prediction = y.iloc[-1] if decreases > 3 else np.average(y[-3:])
    print (first_prediction, k)

    submission['target'][i:i+pred_period] = get_prediction(first_prediction, k, pred_period)

submission.to_csv('beat_benchmark_2.csv',index=False)
print('submission file created')