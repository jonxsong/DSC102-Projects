"""
Brent Min
Jon Zhang
DSC102 ~ PA1
"""
import dask
import ast
from dask.distributed import Client
import dask.dataframe as dd
import numpy as np
import pandas as pd
import json

def get_super_category(categories):
    """
    This helper method will return the super category
    """
    try:
        return ast.literal_eval(categories)[0][0]
    except ValueError:
        return ''

def get_related(related):
    """
    This helper method will return all of the related values into a list
    """
    try:
        answer = np.array([])
        related_dict = ast.literal_eval(related)
        for keys,values in related_dict.items():
            answer = np.append(answer, values)
        return answer
    except ValueError:
        return np.NaN

def Assignment1B(user_reviews_csv,products_csv):
    client = Client('127.0.0.1:8786')
    client = client.restart()

    # defining data types
    reviews_dtypes = {'reviewerID': np.str,
                      'asin': np.str,
                      'reviewerName': np.str,
                      'helpful': np.object,
                      'reviewText': np.str,
                      'overall': np.float64,
                      'summary': np.str,
                      'unixReviewTime': np.float64,
                      'reviewTime': np.str}

    products_dtypes = {'asin': np.str,
                       'salesRank': np.object,
                       'imUrl': np.str,
                       'categories': np.object,
                       'title': np.str,
                       'description': np.str,
                       'price': np.float64,
                       'related': np.object,
                       'brand': np.str}

    # instantiating dataframes as variables
    products = dd.read_csv(products_csv, dtype=products_dtypes)
    reviews = dd.read_csv(user_reviews_csv, dtype=reviews_dtypes)

    ### Question 1 ###

    # percentage of missing values for all columns in the reviews table and the products table
    products_missing_perc = np.mean(products.isnull()) * 100
    reviews_missing_perc = np.mean(reviews.isnull()) * 100

    ### Question 2 ###

    # using only the columns we need to join on
    reviews_sub = reviews[['asin', 'overall']]
    products_sub = products[['asin', 'price']]

    # declaring types for no typeerrors
    reviews_sub['asin'] = reviews_sub['asin'].astype(str)
    products_sub['asin'] = products_sub['asin'].astype(str)

    # joining the dataframes and calculating the pearson correlation
    merged_df = dd.merge(products_sub, reviews_sub, on='asin')
    pearson_correlation = merged_df[['price', 'overall']].corr()
    pearson_correlation = pearson_correlation['price']

    ### Question 3 ###

    # calculating the descriptive statistics
    descriptive_stats = products['price'].describe()

    ### Question 4 ###

    # aggregating over the categories column
    super_category = products['categories'].apply(get_super_category, meta='str').value_counts()

    # parallelizing the individual questions
    q1a, q1b, q2, q3, q4, product_asin = dd.compute(products_missing_perc,
                                                    reviews_missing_perc,
                                                    pearson_correlation,
                                                    descriptive_stats,
                                                    super_category,
                                                    products.asin)

    # converting each question to the correct format for writing into json
    q1a = q1a.round(2).to_dict()
    q1b = q1b.round(2).to_dict()
    q2 = q2['overall'].round(2)
    q3 = q3.round(2)[['mean', 'std', '50%', 'min', 'max']].to_dict()
    q4 = q4.to_dict()

    ### Question 5 ###

    # check if the review ids are in the computed product ids
    product_is_not_dangling = reviews.asin.isin(product_asin)
    if all(product_is_not_dangling) == True:
        q5 = 0
    else:
        q5 = 1

    ### Question 6 ###

    # extract just the related column as a dataframe
    products_related = products[['related']]

    # aggregate over just the related column as a series
    products_related['related'] = products_related.related.apply(get_related, meta='array')

    # get the list of product ids separated into individual values using .explode()
    asins = products_related.explode('related')

    # check if the list of product ids are in the computed product ids
    asin_is_not_dangling = asins.related.isin(product_asin)
    if all(asin_is_not_dangling) == True:
        q6 = 0
    else:
        q6 = 1

    # correct format according to PA1 writeup
    submit = {'q1': {'products': q1a, 'reviews': q1b},
              'q2': q2,
              'q3': q3,
              'q4': q4,
              'q5': q5,
              'q6': q6}

    with open('results_PA1.json', 'w') as outfile: json.dump(submit, outfile)

if __name__ == '__main__':
    Assignment1B('user_reviews.csv', 'products.csv')
