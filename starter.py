import pandas as pd
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Splits the year_month column into year and month and returns them as integers
def get_year_month(year_month):
    years = []
    months = []
    for y_m in year_month:
        y,m = y_m.split('/')
        years.append(int(y))
        months.append(int(m))
    return years,months

def get_zero_repairs(st_year,end_year):
    zero_df = []
    years = list(range(st_year,end_year +1))
    months = list(range(1,13))
    zero_df = [(y,m,0) for y,m in itertools.product(years,months)]
    zero_df = pd.DataFrame(zero_df)
    zero_df.columns = ['year','month','number_repair']
    return zero_df

# Return repair data for a specific module and component
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

# Given an initial prediction first_prediction
# Apply decay coefficient k for num_preds number of periods 
def get_prediction(first_prediction, k, num_preds):
    result = [first_prediction]

    for i in range(1, num_preds):
        result.append(result[-1] * k)

    return result

repair_train = pd.read_csv('../data/RepairTrain.csv')
repair_train['year_sale'],repair_train['month_sale'] = get_year_month(repair_train['year/month(sale)'])
repair_train['year_repair'],repair_train['month_repair'] = get_year_month(repair_train['year/month(repair)'])
repair_min_year = repair_train['year_repair'].min()
repair_max_year = repair_train['year_repair'].max()

cols_requ = ['module_category','component_category','year_repair','month_repair','number_repair']
cols_groupby = ['module_category','component_category','year_repair','month_repair']
repair_train_summ = repair_train[cols_requ].groupby(cols_groupby).sum()

output_target = pd.read_csv('../data/Output_TargetID_Mapping.csv')
submission = pd.read_csv('../data/SampleSubmission.csv')

# Number of predictions to make
pred_period = 19

c = 0
# Make predictions for each module-component
for i in range(0,output_target.shape[0],pred_period):
    module = output_target['module_category'][i]
    category = output_target['component_category'][i]
    print 'Predicting for ' + str(module) + ' ' + str(category) + '.'

    # Replace all empty fields with 0
    # Use only the last 8 months to predict
    X = get_repair_complete(module,category).fillna(0).query('year >= 2009 & month >= 5')
    
    # Scale first month to 0
    years = X.year.apply(lambda y: (y-2009)*12)
    months = X.month.apply(lambda m: m - 5)
    x = years.astype(int).combine(months, func=lambda x, y: x + y)[:, np.newaxis]
    y = X.number_repair

    # Take the log of (number of repairs + 1) for the last 8 months
    # Fit a line through the resulting datapoints
    lr = LinearRegression()
    lr.fit(x, y.apply(lambda y: np.log(y + 1)))

    # Calculate decay coefficient k
    k = lr.coef_[0] if lr.coef_[0] <= 0 else np.log(0.91)
    k = np.exp(k)

    # Initial month (01/2010) prediction is same as previous month (12/2009)
    first_prediction = y.iloc[-1]

    if c == 3:
        plt.plot(x, y, 'ko')
        plt.show()

    # Make predictions
    submission['target'][i:i+pred_period] = get_prediction(first_prediction, k, pred_period)
    c += 1

# submission.to_csv('submission.csv',index=False)

print 'submission.csv created.'







