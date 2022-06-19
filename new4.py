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

    SELECT * 

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

    #remove duplicated columns
    df = df.loc[:, ~ df.columns.duplicated()]

    #drop unnecessary/redundant columns
    df.drop(columns = ['transactiondate', 'parcelid', 'id', 'heatingorsystemtypeid', 'propertyzoningdesc', 'buildingqualitytypeid', 
    'censustractandblock', 'propertycountylandusecode', 'calculatedbathnbr', 
    'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemdesc'], axis = 1, inplace = True)

    #create cached csv
    #df.to_csv('zillow.csv', index = False)                          
    return df



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def handle_missing_values(df, prop_required_column, prop_required_row):
    
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis = 0, thresh = n_required_row)
    df = df.dropna(axis = 1, thresh = n_required_column)
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    #df_dummies = df_dummies.drop(columns = ['regionidcounty'])
    return df_dummies
   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_features(df):
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])

    # create taxrate variable
    df['tax_rate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200], 
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                       )

    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet


    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                            )


    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                      )


    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})


    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    # 12447 is the ID for city of LA. 
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prep_zillow(df):

    #use function to get encoded counties
    df = get_counties(df)

    #use function to create new features
    df = create_features(df)

    #use function to handle missing data to drop columns/rows that have 50% missing values
    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .5)

    #use function to discard outliers
    df = discard_outliers(df, 1.5, ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'taxamount', 'lotsizesquarefeet', 'logerror'])

    #fill unitcnt w/ mode
    df['unitcnt'] = df['unitcnt'].fillna(df['unitcnt'].mode()[0])

    #only maintain unitcnt == 1
    df = df[df.unitcnt == 1.0]

    #fill yearbuilt with mode
    df['yearbuilt'] = df['yearbuilt'].fillna(df['yearbuilt'].mode()[0])


    #add column for binned logerrors
    df['logerror_bins'] = pd.cut(df.logerror, [-5, -.2, -.05, .05, .2, 4])

    #rename df columns
    df = df.rename(columns = {
                              'bathroomcnt': 'bathrooms',
                              'bedroomcnt': 'bedrooms',
                              'calculatedfinishedsquarefeet': 'area',
                              'fips': 'county',
                              'lotsizesquarefeet': 'lot_size',
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
                              'landtaxdollarcnt': 'total_land_tax',
                              'landtaxvaluedollarcnt': 'land_value',
                              'taxvaluedollarcnt': 'total_tax',
                              'taxamount': 'tax_amount',
                              'max_transactiondate': 'transaction_date',
                              'propertylandusedesc': 'property_desc'})

    #change fips to categorical using map to show county info:
    df['county'] = df.county.map({6037.0: 'LA', 6059.0: 'OC', 6111.0: 'VC'})

    #undo 10e6 that was applied to lat and long
    df[['latitude', 'longitude']] = (df[['latitude', 'longitude']]) / (10 ** 6)

    #undo 10e6 that was applied to census_tract
    df['census_tract'] = (df['census_tract']) / (10 ** 6)

    #create new column for bed/bath
    df['bed_and_bath'] = df['bedrooms'] + df['bathrooms']

    #fill nulls with zero
    df['tax_value'] = df['tax_value'].fillna(0)

    #create new column to bin tax value
    df['tax_value_binned'] =  pd.qcut(df['tax_value'], 3, labels = ['Low', 'Med', 'High'], precision = 2)

    #encode tax_value after dropping nulls
    df['tax_value_encoded'] = df['tax_value_binned'].map({'Low': 0, 'Med': 1, 'High': 2}).astype(int)

    #create new column that creates a boolean for homes built after 1945
    df['pw_build'] = df.year_built > 1945

    #encode pw_build 
    df['pw_build_encoded'] = df['pw_build'].map({False: 0, True: 1}).astype(int)


    df['zip_code'] = df['zip_code'].fillna(df['zip_code'].mode()[0])

    df['region_id_county'] = df['region_id_county'].fillna(df['region_id_county'].mode()[0])

    # convert column to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    #create column for absolute log error
    df.loc[:,'abs_logerror'] = df['logerror'].abs()

    #drop remaining null values
    df = df.dropna()

    df['region_id_city'] = df['region_id_city'].astype(int)

    df['zip_code'] = df['zip_code'].astype(int)

    #train, test, split
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)

    return train, validate, test, df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def wrangled_zillow():
    
    train, validate, test, df = prep_zillow(acquire_zillow())

    return train, validate, test, df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#get scaled data


def scale_data(train, 
               validate, 
               test, 
               columns_to_scale = ['bedrooms', 'bathrooms', 'area', 'year_built', 'lot_size', 'latitude', 'longitude', 
               'zip_code', 'bed_and_bath', 'assessment_year', 'tax_value', 'room_count', 'unit_count', 'tax_rate', 
               'land_dollar_per_sqft', 'census_tract', 'age_bin'],
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
                    columns_to_scale =  ['bedrooms', 'bathrooms', 'area', 'year_built', 'lot_size', 'latitude', 'longitude', 
                    'zip_code', 'bed_and_bath', 'assessment_year', 'tax_value', 'room_count', 'unit_count', 'tax_rate', 
                    'land_dollar_per_sqft', 'census_tract', 'age_bin'],
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

        