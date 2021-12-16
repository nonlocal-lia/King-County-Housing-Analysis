import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

def create_heatmap(df)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5, 8))
    sns.heatmap(
        data=corr, 
        mask=np.triu(np.ones_like(corr, dtype=bool)), 
        ax=ax, 
        annot=True, 
        cbar_kws={"label": "Correlation", "orientation": "horizontal", "pad": .2, "extend": "both"}
        )
    ax.set_title("Heatmap of Correlation Between Attributes (Including Target)")
    return plt.show()

