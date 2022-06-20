import os
from env import get_db_url
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def acquire_mall(use_cache = True):
    if os.path.exists('mall_customers.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('mall_customers.csv')
    print('Acquring from SQL database')
    url = get_db_url('mall_customers')
    query = '''
            
    SELECT * 
    FROM customers
    
    '''

     #create df
    df = pd.read_sql(query, url)

    #create cached csv
    df.to_csv('mall_customers.csv', index = False)                          
    return df
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def one_hot_encode(df):
    df['is_female'] = df.gender == 'Female'
    df = df.drop(columns ='gender')
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split(df):
    train_and_validate, test = train_test_split(df, random_state = 123, test_size = .2)
    train, validate = train_test_split(train_and_validate, random_state = 123, test_size = .3)

    print('Train: %d rows, %d cols' % train.shape)
    print('Validate: %d rows, %d cols' % validate.shape)
    print('Test: %d rows, %d cols' % test.shape)

    return train, validate, test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def scale(train, validate, test):

    columns_to_scale = ['age', 'spending_score', 'annual_income']
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])

    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    return train_scaled, validate_scaled, test_scaled

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_exploration_data():
    df = acquire_mall()
    train, validate, test = split(df)
    return train

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_modeling_data(scale_data = False):
    df = acquire_mall()
    df = one_hot_encode(df)

    train, validate, test = split(df)

    if scale_data:
        return scale(train, validate, test)
    else:
        return train, validate, test

