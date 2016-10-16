from collections import defaultdict
import pandas as pd
import pickle


def ddd():
    return defaultdict(dict)
reviewer_dict = defaultdict(ddd)
product_dict = defaultdict(ddd)

relevant_columns = ['reviewTime', 'helpful_fraction', 'helpful_number_of_ratings', 
                    'count_negative', 'count_positive', 'overall', 'spanish_count', 
                    'counts_positive_negative_good_count', 'counts_rating_bad_count', 
                    'counts_positive_negative_weighted_bad_count', 'counts_helpful_bad_count',
                    'counts_rating_good_count', 'counts_helpful_good_count', 
                    'counts_positive_negative_bad_count', 
                    'counts_positive_negative_weighted_good_count', 'count_words']

reviews = pd.read_csv('../amazon/tmp/review_features.csv')
reviews['reviewTime'] = reviews['unixReviewTime'].apply(lambda x: (x - 929232000)/(1406073600-929232000) * 
                                                        (2014.47671232 - 1999.45205479) + 1999.45205479)

for index in reviews.index:
    for col in relevant_columns:
        reviewer_dict[reviews.ix[index, 'reviewerID']][reviews.ix[index, 'reviewID']][col] = reviews.ix[index, col]
        
    for col in relevant_columns:
        product_dict[reviews.ix[index, 'asin']][reviews.ix[index, 'reviewID']][col] = reviews.ix[index, col]
    
print('The number of individual reviewers is {}'.format(len(reviewer_dict)))
print('The number of individual products is {}'.format(len(product_dict)))
    
pickle.dump(reviewer_dict, open('../amazon/tmp/reviewer_dict.pickle', 'wb'))
pickle.dump(product_dict, open('../amazon/tmp/product_dict.pickle', 'wb'))