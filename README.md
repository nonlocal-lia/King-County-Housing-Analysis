![Header Image](/images/king_county_photo.jpeg)

# King County Home Renovation Analysis

Author: [Lia Elwonger](mailto:lelwonger@gmail.com)

## Overview

This project analyses King County home sales data from 2014 to 2015 to construct a model of the King COunty housing market for a hypothetical real estate business
looking for recommendations for clients seeking potential renovate their home prior to a sale. 

### Business Problem

The median real estate agency has a number of significant business problems. One is how to advise clients hoping to sell their homes 
on what they can do to maximize the price of their sale. Information on the likely payoff of renovating their homes could be useful to many clients in
making a decision as to whether a potential renovtion is worth the cost and time investment.

To aid in making these recommendations we will attempt to answer three questions:

* How much would renovating to improve the condition or grade of the home likely increase the sale price?
* How much would adding an extension to the living area of the home likely increase sale price?
* How much would adding an additional bathroom likely increase sale price?

### The Data

This project primarily uses the King County House Sales dataset, which can be found in  `kc_house_data.csv` in the data folder in this project's GitHub repository. The description of the column names can be found in `column_names.md` in the same folder. Data on neighborhood names for various zipcodes in King County can also be found in `king_county_neighborhoods.csv` in the data folder and was used in the construction of the final model.

### Modeling

#### Preperation
Categorical data including condition, grade, number of bathrooms, bedrooms and floors were one-hot encoded.
Data on years of renovation was binned into recent, old and never renovated categories, with 10 years counting as the threshold for a recent renovation.
Given how few homes had basements, data on the sqft of basements was binned into categories representing the presence or absence of a basement.
The age of the homes were added as a feature to the data using the year of construction.
Zipcodes were used along with data from 'king_county_neighborhoods.csv' to categorize homes into rough cities/neighborhoods which were then one-hot encoded.
The age of the homes were added as a feature to the data using the year of construction.
Non-categorical data such as the price, and sqft of homes and lots were log normalized to improve the performance of the model.

#### Method of Modeling
A multiple OLS regression model with a target variable of home prices was constructed using variables representing: square footage of the home and lot, square footage of the neighboring 15 homes and lots, number of bedrooms, bathrooms, and floors in each home, house grade, house condition, and neighborhood of the home

### Results

The final regression had an R-squared value of 0.85.

Here is a visualization of the performance of the model with some test data:

![prediction](/images/prediction_graph.png)

Using the coefficients from the model we can make predictions about the relative effects of various changes to the home.

Condition predictions:

![condition](/images/condition.png)

Grade predictions:

![grade](/images/grade.png)

Renovation predictions:

![renovation](/images/renovation.png)

Bathroom predictions:

![bathrooms](/images/bathrooms.png)

Living Area predictions:

![square_footage](/images/sqft.png)

Some limitations of the model to note are that the underlying data had a high kurotosis of 4.984 even after the data was log normalized, this impacts the performance of the model and there was signficant multicollinearity between the square footage variables and the numbers of bathrooms, bedrooms and floors, which will impact the usefulness of the coefficients of the model in making inferences about those features.

### Conclusions

When advising your clients, you can give the following pieces of advice:
    
* Increasing the grade score of a home by one level will increase the home value by around 10% for grades less than 8 and by about 20% for higher grades.
* Renovating an unrenovated home will typically increase its price around 14%, but only by around 2% if it has been previously renovated.
* Increasing the square footage of the home will  increase the home price by about 0.44% per 1% of added space.
* Adding an additional bathroom may increase the price by around 5% but is highly dependent on context.

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
