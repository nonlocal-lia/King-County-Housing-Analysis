import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def create_heatmap(df, size=(12,12)):
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

def linearity_graph(model, X_test, y_test):
    preds = model.predict(X_test)
    fig, ax = plt.subplots()
    perfect_line = np.arange(y_test.min(), y_test.max())
    ax.plot(perfect_line, linestyle="--", color="orange", label="Perfect Fit")
    ax.scatter(y_test, preds, alpha=0.5)
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.legend()
    return plt.show()

def normality_graph(model, X_test, y_test):
    preds = model.predict(X_test)
    residuals = (y_test - preds)
    sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
    return plt.show()

def homoscedasticity_graph(model, X_test, y_test):
    preds = model.predict(X_test)
    residuals = (y_test - preds)
    fig, ax = plt.subplots()
    ax.scatter(preds, residuals, alpha=0.5)
    ax.plot(preds, [0 for i in range(len(X_test))])
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Actual - Predicted Value")
    return plt.show()
