import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def create_heatmap(df, size=(12,12)):
    """
    Produces a correlation heat map from a panda dataframe

    Arg:
        df(pdDataFrame): a dataframe to plot
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays heat map plot
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(
        data=corr, 
        mask=np.triu(np.ones_like(corr, dtype=bool)), 
        ax=ax, 
        annot=True, 
        cbar_kws={"label": "Correlation", "orientation": "horizontal", "pad": .2, "extend": "both"}
        )
    ax.set_title("Heatmap of Correlation Between Attributes (Including Target)")
    return plt.show()

def linearity_graph(model, X_test, y_test, size=(12,12)):
    """
    Produces a linearity test scatter plot from a panda dataframe

    Arg:
        model: a Scikit learn OLS model constructed for the variables input
        X_test(pdDataFrame): a dataframe wih the independent variables to test the model on
        y_test(pdDataFrame): a dataframe wih the dependent variable to test the model on
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the linearity plot
    """

    preds = model.predict(X_test)
    fig, ax = plt.subplots(figsize=size)
    perfect_line = np.arange(y_test.min(), y_test.max())
    ax.plot(perfect_line, perfect_line, linestyle="--", color="orange", label="Perfect Fit")
    ax.scatter(y_test, preds, alpha=0.5)
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.legend()
    return plt.show()

def normality_graph(model, X_test, y_test, size=(12,12)):
    """
    Produces a Q-Q plot from a panda dataframe

    Arg:
        model: a Scikit learn OLS model constructed for the variables input
        X_test(pdDataFrame): a dataframe wih the independent variables to test the model on
        y_test(pdDataFrame): a dataframe wih the dependent variable to test the model on
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the normality plot
    """
    fig, ax = plt.subplots(figsize=size)
    preds = model.predict(X_test)
    residuals = (y_test - preds)
    sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
    return plt.show()

def homoscedasticity_graph(model, X_test, y_test, size=(12,12)):
    """
    Produces a homoscedasticity test scatter plot from a panda dataframe

    Arg:
        model: a Scikit learn OLS model constructed for the variables input
        X_test(pdDataFrame): a dataframe wih the independent variables to test the model on
        y_test(pdDataFrame): a dataframe wih the dependent variable to test the model on
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the homoscedasticity plot
    """
    fig, ax = plt.subplots(figsize=size)
    preds = model.predict(X_test)
    residuals = (y_test - preds)
    ax.scatter(preds, residuals, alpha=0.5)
    ax.plot(preds, [0 for i in range(len(X_test))], linestyle="--", color='orange', label="Perfect Fit")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Actual - Predicted Value")
    return plt.show()

def one_hot_coef_graph(coef_df, categories, dropped_var, target_name='Price', increase_type = 'Percent', size=(12,12)):
    """
    Produces a bar graph of the coefficients from a panda dataframe

    Arg:
        coef_df(pdDataFrame): a dataframe containing the coefficient to graph
        categories(array): a list of the category names whose coefficients you wish to graph
        dropped_var(str): the name of the variable droped during one-hot encoding
        target_name(str): the name of the target variable
        increase_type(str): the type of increase represented by the coefficient, either 'Percent' or 'Absolute'
        size(tuple): a 2 element tuple with the size of the produced figure

    Return:
        Displays the bar graph of the listed categories coefficients
    """
    coefs = [] 
    for cat in categories:
        coefs.append(float(coef_df[cat]))
    fig, ax = plt.subplots(figsize=size)
    ax = plt.bar(categories, coefs)
    plt.ylabel("{x} Increase in {y}".format(x=increase_type, y=target_name))
    plt.title("Predicted {x} Increase in {y} Relative to {x}".format(x=increase_type, y=target_name, z=dropped_var))
    return plt.show()