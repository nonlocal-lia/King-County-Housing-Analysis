![Header Image](/images/king_county_photo.jpeg)

# King County Home Renovation Analysis

Author: [Lia Elwonger](mailto:lelwonger@gmail.com)

## Overview

This project analyses King County home sales data from 2014 to 2015 to construct a model of the housing market for hypothetical busiess purposes. 

### Business Problem

The median real estate agency has a number of significant business problems. One is how to advise clients hoping to sell their homes 
on what they can do to maximize the price of their sale. Information on the likely payoff of renovating their homes could be useful to many clients in
making a decision as to whether a potential renovtion is worth the cost and time investment.

To aid in making these recommendations we will attempt to answer three questions:

* How much would renovating to improve the condition or grade of the home likely increase the sale price?
* How much would adding an extension to the living area of the home likely increase sale price?
* How much would adding an additional bathroom likely increase sale price?

### The Data

This project primarily uses the King County House Sales dataset, which can be found in  `kc_house_data.csv` in the data folder in this project's GitHub repository. The description of the column names can be found in `column_names.md` in the same folder. Data on neighborhood names for various zipcodes in King County can also be found in 'king_county_neighborhoods.csv' in the data folder and was used in the construction of the final model.

### Modeling

#### Preperation
Clearly categorical data including condition and grade were one-hot encoded.
Data on years of renovation was binned into recent, old and never renovated categories, with 10 years counting as the threshold for a recent renovation.
Given how few homes had basements, data on the sqft of basements was binned into categories representing the presence or absence of a basement.
The age of the homes were added as a feature to the data using the year of construction.
Zipcodes were used along with data from 'king_county_neighborhoods.csv' to categorize homes into rough cities/neighborhoods which were then one-hot encoded.
The age of the homes were added as a feature to the data using the year of construction.
Data on the number of bathrooms, bedrooms, and floors were treated as oridinal label encoded data.
Non-categorical data such as the price, and sqft of homes and lots were log normalized to improve the performance of the model.

#### Method of Modeling
A multiple OLS regression model with a target variable of home prices was constructed using variables representing: square footage of the home and lot, square footage of the neighboring 15 homes and lots, number of bedrooms, bathrooms, and floors in each home, presence or absence of a basement, house grade, house condition, house age and neighborhood of the home

### Results

The final regression had an R-squared value of 0.85 and a mean-squared error of _

*Image of Mean square error*

*image of condition/grade predictions*

*image of living area predictions*

*image of bathroom predictions*

### Conclusions

* Tell your client X
* Tell your client Y
* Tell your client Z

### Limitations and Future Work

Technically all this analysis can tell us is the association between renovations and certain house features and prices, it can't actually tell us
whether renovating a particular home would lead to a return. With a larger dataset covering more years it might be possible to do an analysis comparing the prices of homes before and after renovations.

The recomendations are also all conditional on the price of making various home renovations. WIth data on what contractors charge for various renovations, it would be possible to produce more solid recommendations from the model.

## For More Information

Please review my full analysis in our Jupyter Notebook or our presentation.

For any additional questions, please contact **Lia Elwonger lelwonger@gmail.com**

## Repository Structure

```
├── README.md                           <- The top-level README for reviewers of this project
├── king-county-housing-analysis.ipynb  <- Narrative documentation of analysis in Jupyter notebook
├── king_county_presentation.pdf        <- PDF version of project presentation
├── data                                <- Both sourced externally and generated from code
├── code                                <- Functions for cleaning and processing the data and constructing visualizations
└── images                              <- Both sourced externally and generated from code
```