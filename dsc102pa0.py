"""
Brent Min  |  A15053745
Jonathan Zhang  |  A13646254

PA0.py is the script for writing in our Dask DataFrame Data Explorations into "results_PA0.json"
"""
from dask.distributed import Client
import dask.dataframe as dd
import numpy as np
import pandas as pd
import json    
    
    
def num_helpful(row):
    """
    This function will return the helpful votes of the column 'helpful'
    """
    try:
        return int(row[1:row.index(',')])
    except ValueError:
        return 0


def total_helpful(row):
    """
    This function will return the total votes of the column 'helpful'
    """
    try:
        return int(row[row.index(' '):-1])
    except ValueError:
        return 0
    
    
def PA0(user_reviews_csv):
    """
    PA0 will output the results of our completed DataFrame into "results_PA0.json"
    """
    client = Client()
    client = client.restart()
    
    # column data types
    dtypes = {
        'reviewerID': np.str,
        'asin': np.str,
        'reviewerName': np.str,
        'helpful': np.object,
        'reviewText': np.str,
        'overall': np.float64,
        'summary': np.str,
        'unixReviewTime': np.float64,
        'reviewTime': np.str
    }
    
    # defining our DataFrame with the columns listed below
    df = dd.read_csv(user_reviews_csv, dtype=dtypes)
    df = df[['reviewerID', 'asin', 'helpful', 'overall','reviewTime']]
    
    # creating the column 'year' -> must be an integer
    df['year'] = df['reviewTime'].str[-4:].astype(int)
    
    # creating the column 'helpful_votes' -> must be an integer
    df['helpful_votes'] = df.helpful.apply(num_helpful, meta='int64').astype('int')
    
    # creating the column 'total_votes' -> must be an integer
    df['total_votes'] = df.helpful.apply(total_helpful, meta='int64').astype('int')
    
    # essentially drops old columns replaced with the columns we just created above
    df = df[['reviewerID', 'asin', 'overall','year','helpful_votes', 'total_votes']]
    
    # aggregate function to apply numpy functions across specified columns
    grouped = df.groupby('reviewerID').agg({'asin': ['count'],
                                            'overall': ['mean'],
                                            'year': ['min'],
                                            'helpful_votes': ['sum'],
                                            'total_votes': ['sum']},
                                             split_out=4)
    
    # drops the extraneous column level created from the aggregate method (names)
    grouped.columns = grouped.columns.droplevel(1)
    
    # renames columns to fit PA0 writeup
    grouped = grouped.rename(columns={'asin' : 'number_products_rated',
                                      'overall' : 'avg_ratings',
                                      'year' : 'reviewing_since'})
    
    submit = grouped.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: json.dump(json.loads(submit.to_json()), outfile)
    
if __name__ == '__main__':
    PA0('user_reviews.csv')
        
        
        
        
        
        
        
        
    