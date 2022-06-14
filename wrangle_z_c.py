
import os
from env import get_db_url

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




def acquire_zillow(use_cache = True):

    '''This function acquires data from SQL database if there is no cached csv and returns it as a dataframe.'''

    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    print('Acquiring from SQL database')

    url = get_db_url('zillow')
    query = '''

    SELECT * , 
    logerror, 
    transactiondate,
    airconditioningdesc,
    architecturalstyledesc,
    buildingclassdesc,
    heatingorsystemdesc,
    storydesc,
    typeconstructiondesc

    FROM properties_2017

    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
        ) pred USING(parcelid)

    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate

    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    LEFT JOIN storytype USING (storytypeid)
    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)


    WHERE propertylandusedesc IN ("Single Family Residential",
                                  "Mobile Home",
                                  "Townhouse",
                                  "Cluster Home",
                                  "Condominium",
                                  "Cooperative",
                                  "Row House",
                                  "Bungalow",
                                  "Manufactured, Modular, Prefabricated Homes",
                                  "Inferred Single Family Residential"

                                )
    AND transactiondate <= "2017-12-31"
    '''
    #create df
    df = pd.read_sql(query, url)

    #create cached csv
    df.to_csv('zillow.csv', index = False)                          
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def overview(df):
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include = 'all'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def nulls_by_columns(df):
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def nulls_by_rows(df):
    return pd.concat([
        df.isna().sum(axis = 1).rename('n_missing'),
        df.isna().mean(axis = 1).rename('percent_missing'),
    ], axis = 1).value_counts().sort_index()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def discard_outliers(df, k, col_list):
    
    for col in col_list:
        #obtain quartiles
        q1, q3 = df[col].quantile([.25, .75]) 
        
        #obtain iqr range
        iqr = q3 - q1
        
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
        
        #return outlier - free df
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#this is required because in this dataset, some columns are missing entire rows, so dropna() results in an empty dataset

def handle_missing_values(df, prop_required_column, prop_required_row):
    
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis = 0, thresh = n_required_row)
    df = df.dropna(axis = 1, thresh = n_required_column)
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def prep_zillow(df):

   

    df.drop(columns = 'transactiondate')

    #rename df columns
    df = df.rename(columns = {
                              'propertylandusetypeid':'prop_land_id',
                              'heatingrorsystemtypeid': 'heat_type_id',
                              'parcelid': 'parcel_id',
                              'bathroomcnt': 'bathrooms',
                              'bedroomcnt': 'bedrooms',
                              'buildingqualitytypeid': 'building_quality_id',
                              'calculatedbathnbr': 'calculated_bath_bed',
                              'calculatedfinishedsquarefeet': 'area',
                              'finishedsquarefeet12': 'finished_liv_area',
                              'fips': 'county',
                              'fullbathcnt': 'full_bath_cnt',
                              'lotsizesquarefeet': 'lot_size',
                              'propertylandusecode': 'prop_land_use_code',
                              'propertzoningdesc': 'prop_zone_desc',
                              'rawcensustractandblock': 'census_tract',
                              'regionidcity': 'region_id_city',
                              'regionidcounty': 'region_id_county',
                              'regionidzip': 'zip_code',
                              'roomcnt': 'room_count',
                              'unitcnt': 'unit_count',
                              'yearbuilt': 'year_built',
                              'structuretaxvaluedollarcnt': 'tax_value',
                              'assessmentyear': 'assessment_year',
                              'landtaxdollarcnt': 'total_tax',
                              'taxamount': 'tax_amount',
                              'censustractandblock': 'census_tract_block',
                              'logerror': 'log_error',
                              'max_transactiondate': 'transaction_date',
                              'heatingorsystemdesc': 'heating_desc',
                              'propertylandusedesc': 'property_desc'})


    #change fips to categorical using map to show county info:
    df['county'] = df.county.map({6037.0: 'LA', 6059.0: 'OC', 6111.0: 'VC'})

    #undo 10e6 that was applied to lat and long
    df[['latitude', 'longitude']] = (df[['latitude', 'longitude']]) / (10 ** 6)

    #undo 10e6 that was applied to census_tract
    df['census_tract'] = (df['census_tract']) / (10 ** 6)

    #create new column for bed/bath
    df['bed_and_bath'] = df['bedrooms'] + df['bathrooms']

    #create new column to bin year_built
    df['year_built_binned'] = pd.cut(x = df['year_built'], 
                                 bins = [1878, 1909, 1919, 1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019], 
                                 labels=[1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])

    #create new column to bin tax value
    #df['tax_value_binned'] =  pd.qcut(df['tax_value'], 3, labels = ['Low', 'Med', 'High'], precision = 2)


    #Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['county']], dummy_na = False, \
                              drop_first = True)

    #Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis = 1)


    #eliminate values that did not occur in 2017
    #df = df[(df.transaction_date <= '2017-12-31')]

    #eliminate lot_size elimination
    df = df[df.lot_size < 200000]

    #drop null values, CANNOT USE HERE!!! 
    #df = df.dropna()

    #drop null values in tax_value_binned only
    #df = df.dropna(subset=['tax_value_binned'], inplace = True)

    #encode tax_value after dropping nulls
    #df['tax_value_encoded'] = df['tax_value_binned'].map({'Low': 0, 'Med': 1, 'High': 2}).astype(int)

    #create new column that creates a boolean for homes built after 1945

    df['pw_build'] = df.year_built > 1945

    #encode pw_build 
    df['pw_build_encoded'] = df['pw_build'].map({False: 0, True: 1}).astype(int)

    #return COLUMNS that are not duplicated
    df = df.loc[:, ~ df.columns.duplicated()]

    #df = df.drop_duplicates()

    #use function to discard outliers
    df = discard_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount', 'lot_size'])

    #use function to handle missing data to drop columns/rows that have 50% missing values
    df = handle_missing_values(df, prop_required_column= .5, prop_required_row= .5)

    #train, test, split
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)

    return train, validate, test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#get scaled data

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale = ['bedrooms', 'bathrooms', 'area', 'year_built', 'lot_size', 'latitude', 'longitude', 'bed_and_bath'],
               return_scaler = False):

    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns = train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns = validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns = test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def wrangled_zillow():
    
    train, validate, test = prep_zillow(acquire_zillow())

    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#split county

def split_county_tvt(train, validate, test):

    train_VC = train[train.county == 'VC'].copy()
    train_LA = train[train.county == 'LA'].copy()
    train_OC = train[train.county == 'OC'].copy()

    validate_VC = validate[validate.county == 'VC'].copy()
    validate_LA = validate[validate.county == 'LA'].copy()
    validate_OC = validate[validate.county == 'OC'].copy()

    test_VC = test[test.county == 'VC'].copy()
    test_LA = test[test.county == 'LA'].copy()
    test_OC = test[test.county == 'OC'].copy()

    return train_VC, train_LA, train_OC, validate_VC, validate_LA, validate_OC, test_VC, test_LA, test_OC    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #get scaled data for split counties

def scale_split_data(train, 
                    validate,
                    test,
                    columns_to_scale = ['bedrooms', 'bathrooms', 'area', 'year_built', 'lot_size', 'latitude', 'longitude', 'bed_and_bath'],
                    return_scaler = False):

    '''
    Scales the 9 data splits. 
    Takes in train, validate, and test data splits from the county splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    train_scaled_VC = train[train.county == 'VC'].copy()
    train_scaled_LA = train[train.county == 'LA'].copy()
    train_scaled_OC = train[train.county == 'OC'].copy()

    validate_scaled_VC = validate[validate.county == 'VC'].copy()
    validate_scaled_LA = validate[validate.county == 'LA'].copy()
    validate_scaled_OC = validate[validate.county == 'OC'].copy()


    test_scaled_VC = test[test.county == 'VC'].copy()
    test_scaled_LA = test[test.county == 'LA'].copy()
    test_scaled_OC = test[test.county == 'OC'].copy()

    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled_VC[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns = train[columns_to_scale].columns.values).set_index([train.index.values])

    train_scaled_LA[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns = train[columns_to_scale].columns.values).set_index([train.index.values])

    train_scaled_OC[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
              
                                                  columns = train[columns_to_scale].columns.values).set_index([train.index.values])

                                                  
    validate_scaled_VC[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns = validate[columns_to_scale].columns.values).set_index([validate.index.values])

    validate_scaled_LA[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns = validate[columns_to_scale].columns.values).set_index([validate.index.values])

    validate_scaled_OC[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns = validate[columns_to_scale].columns.values).set_index([validate.index.values])



    test_scaled_VC[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns = test[columns_to_scale].columns.values).set_index([test.index.values])
    
    test_scaled_LA[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns = test[columns_to_scale].columns.values).set_index([test.index.values])

    test_scaled_OC[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns = test[columns_to_scale].columns.values).set_index([test.index.values])

    if return_scaler:
        return scaler, train_scaled_VC, train_scaled_LA, train_scaled_OC, validate_scaled_VC, validate_scaled_LA, validate_scaled_OC, test_scaled_VC, test_scaled_LA, test_scaled_OC
    else:
        return train_scaled_VC, train_scaled_LA, train_scaled_OC, validate_scaled_VC, validate_scaled_LA, validate_scaled_OC, test_scaled_VC, test_scaled_LA, test_scaled_OC

        