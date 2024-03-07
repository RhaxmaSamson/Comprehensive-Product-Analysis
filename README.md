# Comprehensive-Product-Analysis
# About the project
Using sales data from a year, the comprehensive product analysis revealed trends in consumer preferences, identifying top-selling items and their respective market segments. Insights gleaned from the analysis will inform strategic decisions, guiding product development and marketing efforts to maximize profitability and market penetration.

# Goals and Objective
  * Goal
    -To identify top-selling products and understand their performance throughout the year.
  * Objective
    -Conduct a thorough analysis of sales data, including volume, revenue, and profit margins for each product.



# Getting Started
To run this project you will need Jupyter notebook or google colab to run data analysis.
## Built with
- Python
- executed in Google Colab
- front end in vscode

## Prerequisites
These are some library you need to run the project, i put the pip installation to make it easy for you.


* Pandas
  pip install pandas
  
* Matplotlib 
  pip install matplotlib

* Seaborn
  pip install seaborn
 

## Resources
Dataset was taken from Kaggle
# Data Processing
Avalaible data are 12 csv files for each month sales data. Dataset were concatenante and resulting 186,850 orders data. Dataset contain 545 null values and some unmatch feature data types. Extract some feature such as month, day, hour, city, sales.
# Data Analysis
## Descriptive Analysis


Customer mostly order 1 item at once, some small group order 2 item at once, highest order are in 9 item at once. Sales for each order are in range 2.99 to 3400. Distribution for sales and Price Each relatively same it is because most quantity order is 1.

Summary sales 2019, total revenue <b>34,483,365.68 USD, 185,916 orders and 209,038 items sold.</b>

## Univariate Analysis
## Mulituvariate Analysis

Top product sold are on Battery products, then followed by Charging cable, and Headphones.

# Insight and Recommendation

1. Product Combination <br>
There are some frequently combination of products in customer orders behavior. Most of combination are in : 


    - Phone product + Charging cable 
    - Phone product + Headphone
    - Charging cable + Headphone

    This data can support to make product bundling to increase more sales of specifict product.

2. Rush Hour <br>
There is peak of sales in around 9:00 to 21:00. It is means that in this time range mostly customer tend to place order. This peak can be sweet spot to promote advertising.

    This data can be support to post more ads on the rush hour time span.

3. Order Probability <br>
Charging cable have relatively same probability. iPhone have higher probability than Google Phone. Wired Headphones have highest order probability on headphones product type.

    This data can support to have more product stock and marketing on higher product order probability.

