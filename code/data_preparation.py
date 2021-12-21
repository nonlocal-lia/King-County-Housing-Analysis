import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate, ShuffleSplit


def z_standardize(x):
    """
    Standardizes array using z-score and produces a sklearn scaler object for standarizing/destandarizing

    Arg:
        x(array): Array to standardize

    Return:
        standard(array): Standardized array 
        scaler: a scaler object from the scikit preprocessing module
    """
    scaler = StandardScaler().fit(x)
    standard = scaler.transform(x)
    return standard, scaler

def destandardize(x, scaler):
    """
    Reverses the standardization of an array using z-score

    Arg:
        x(array): Array to destandardize
        scaler: a scaler object from the scikit preprocessing module

    Return:
        destandard(array): Non-standardized array 
    """
    destandard = scaler.inverse_transform(x)
    return destandard


def log_normalize(df, columns, plus_1 = True):
    """
    Transforms column values into log values and normalizes them using z-scores and produces a sklearn scaler object for standarizing/destandarizing

    Arg:
        df(pdDataFrame): A dataframe with values to normalize
        columns(array): An array containing column names of the columns to normalize

    Return:
        df_norm(pdDataFrame): A dataframe with normalized log values in the specified columns
        scaler: a scaler object from the scikit preprocessing module
    """
    
    if plus_1:
        transformer = FunctionTransformer(np.log1p, validate=True, inverse_func=np.expm1, check_inverse=True)
        df_log = transformer.transform(df[columns])
        df_norm, scaler = z_standardize(df_log)
        df = pd.DataFrame(df_norm)
        df.columns = columns
        return df, scaler
    transformer = FunctionTransformer(np.log, validate=True, inverse_func=np.expm1, check_inverse=True)
    df_log = transformer.transform(df[columns])
    df_norm, scaler = z_standardize(df_log)
    df = pd.DataFrame(df_norm)
    df.columns = columns
    return df, scaler

def log_denormalize(df, columns, scaler, plus_1 = True):
    """
    Transforms column values that we log transformed and standardized using z-scores back to their original values.

    Arg:
        df(pdDataFrame): A dataframe with values to denormalize
        columns(array): An array containing column names of the columns to denormalize

    Return:
        df_norm(pdDataFrame): A dataframe with denormalized values in the specified columns
    """
    if plus_1:
        df_denorm = destandardize(df[columns], scaler)
        transformer = FunctionTransformer(np.log1p, validate=True, inverse_func=np.expm1, check_inverse=True)
        df_delog = transformer.inverse_transform(df_denorm)
        df = pd.DataFrame(df_delog)
        df.columns = columns
        return df
    df_denorm = destandardize(df[columns], scaler)
    transformer = FunctionTransformer(np.log, validate=True, inverse_func=np.exp, check_inverse=True)
    df_delog = transformer.inverse_transform(df_denorm)
    df = pd.DataFrame(df_delog)
    df.columns = columns
    return df


def missing_indicator(df, column):
    """
    Produces an array of booleans representing missing values from column

    Arg:
        df(pdDataFrame): A dataframe with a column to create a missing indicator array from
        column(str): A string which is the column label of the desired column
    
    Return:
        missing(array): A numpy array containing booleans coresponding to the null values of the column
    """
    c = df[[column]]
    miss = MissingIndicator()
    miss.fit(c)
    missing = miss.transform(c)
    return missing

def impute_values(df, column, type='median'):
    """
    Produces an array of booleans representing missing values from column

    Arg:
        df(pdDataFrame): A dataframe with a column to impute missing values to
        column(str): A string which is the column label of the desired column
        type(str): A name of a method to use to impute values from the sklearn SimpleImputer
    
    Return:
        imputed(array): A numpy array containing imputed values for the missing data
    """
    c = df[[column]]
    imputer = SimpleImputer(strategy=type)
    imputer.fit(c)
    imputed = imputer.transform(c)
    return imputed

def oridinal_encode(df, column):
    """
    Produces an array of category names and an array of integers representing the categories

    Arg:
        df(pdDataFrame): A dataframe with a column of categorical data to encode
        column(str): A string which is the column label of the desired column
    
    Return:
        cat(array): A numpy array containing strings of the categories in the column
        encoded(array): A numpy array containing intergers representing the categories
    """
    c = df[[column]]
    encoder = OrdinalEncoder()
    encoder.fit(c)
    cat = encoder.categories_[0]
    encoded = encoder.transform(c)
    encoded = encoded.flatten()
    return cat, encoded

def one_hot_encode(df, column):
    """
    Produces an array of category names and an array of array contain boolean representing the categories

    Arg:
        df(pdDataFrame): A dataframe with a column of categorical data to encode
        column(str): A string which is the column label of the desired column
    
    Return:
        cat(array): A numpy array containing strings of the categories in the column
        encoded(array): A numpy array containing arrays containing booleans representing the categories
    """
    c = df[[column]]
    ohe = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")
    ohe.fit(c)
    cat = ohe.categories_
    cat = list(cat[0])
    encoded = ohe.transform(c)
    return cat, encoded

def collinearity_check(df, min=0.75, max=1, exclude_max=True):
    """
    Produces a dataframe listing all the correlations between the variables in the dataframe
    in between the the input min and max values

    Arg:
        df(pdDataFrame): A dataframe with a column of categorical data to encode
        min(float64): A float between 0 and 1
        max(float64): A float between 0 and 1, usually 1
    
    Return:
        cat(array): A numpy array containing strings of the categories in the column
        encoded(array): A numpy array containing arrays containing booleans representing the categories
    """
    check = df.corr().abs().stack().reset_index().sort_values(0, ascending=False)
    check['pairs'] = list(zip(df.level_0, df.level_1))
    check.set_index(['pairs'], inplace = True)
    check.drop(columns=['level_1', 'level_0'], inplace = True)
    check.columns = ['cc']
    if exclude_max:
        return check[(check.cc >= min) & (check.cc < max)]
    return check[(check.cc >= min) & (check.cc <= max)]

def mean_square_error(model, X_train, y_train, X_test, y_test):
    """
    Gives the mean of the R-squared values for the model for the specified number of Kfolds

    Arg:
        model: Linear regression model produced from sklearn LinearRegression
        X_train(pdDataFrame): A dataframe containing all the predictor variables the model was trained on
        y_train(pdSeries): A series containing the target varible the model was trained on
        X_test(pdDataFrame): A dataframe containing all the predictor variables the model will be tested against
        y_test(pdSeries): A series containing the target varible the model will be tested against

    Return:
        Prints both training and test mse
        train_mse(float64): A float representing the mean square error of the model relative to the training data
        test_mse(float64): A float representing the mean square error of the model relative to the testing data
    """
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_hat_train)
    test_mse = mean_squared_error(y_test, y_hat_test)
    print('Train Mean Squarred Error:', train_mse)
    print('Test Mean Squarred Error:', test_mse)
    return train_mse,test_mse

def cross_val(model, X_train, y_train, splits=5, test_size=0.25, random_state=0):
    """
    Gives the mean of the R-squared values for the model for the specified number of Kfolds

    Arg:
        model: Linear regression model produced from sklearn LinearRegression
        predictors(pdDataFrame): A dataframe containing all the predictor variables from the model
        target(pdSeries): A series containing the target varible of the model
        fold(int64): The number of Kfolds to perform in the cross validation

    Return:
        Prints mean of the training and test R-squared scores
        scores(dict): A dictionary containing arrays for the following keys: fit_time, score_time, test_score, train_score
    """
    splitter = ShuffleSplit(n_splits=splits, test_size=test_size, random_state=random_state)
    scores = cross_validate(estimator=model, X=X_train, y=y_train, return_train_score=True, cv=splitter)
    print("Train score:     ", scores["train_score"].mean())
    print("Validation score:", scores["test_score"].mean())
    return scores

def predict_median_effect(df, variable_column, scaled_columns, scaler, model, target = 'price'):
    median_df = df.groupby(variable_column).median().reset_index()
    median_df.insert(0, target, model.predict(median_df))
    predict_df = median_df[scaled_columns]
    predict_df = log_denormalize(predict_df, predict_df.columns, scaler, plus_1 = True)
    return predict_df

def predict_difference(model, data, variable, low, high, scaler, scaled_cols, target='price'):
    z = list(zip(scaler.mean_, scaler.scale_))
    scale_dict = dict(zip(scaled_cols, z))
    if variable in scaled_cols:
        standard_low = (np.log1p(low)-scale_dict[variable][0])/scale_dict[variable][1]
        standard_high = (np.log1p(high)-scale_dict[variable][0])/scale_dict[variable][1]
    else:
        standard_low = low
        standard_high = high
    low_variables = data.groupby(variable).median().reset_index()
    low_variables = low_variables[low_variables[variable] == standard_low]
    low_prediction = model.predict(low_variables)
    high_variables = data.groupby(variable).median().reset_index()
    high_variables = high_variables[high_variables[variable] == standard_high]
    high_prediction = model.predict(high_variables)
    raw_difference = high_prediction - low_prediction
    difference = np.expm1(raw_difference*scale_dict[target][1] + scale_dict[target][0])
    return float(difference)