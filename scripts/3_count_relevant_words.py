from collections import defaultdict
import gc
from joblib import Parallel, delayed
import numpy
import os
import pandas as pd
import pickle

num_cores = int(os.environ['ML_NUM_CORES'])


def get_fraction_difference(x):
    """
    return the difference between the fractions of helpful and unhelpful ratings
    """
    x = x[1:-1].split(',')
    if int(x[1]) == 0:
        return numpy.nan
    return -1 + 2 * float(x[0]) / float(x[1])
    
reviews = pd.read_csv('../amazon/tmp/reviews_electronics_no_text.csv')
reviews['helpful_fraction'] = reviews['helpful'].apply(lambda x: get_fraction_difference(x))
# the following weights will account for unequal distributions so that words that simply occur often in the english
# language won't have a major advantage
a_priori_good_bad_overall = (reviews['overall'] < 3).sum() / (reviews['overall'] > 3).sum()
a_priori_helpful_overall = (reviews['helpful_fraction'] < 0).sum() / (reviews['helpful_fraction'] > 0).sum()

lemmatized = pd.concat((pd.read_csv('../amazon/tmp/reviews_electronics_block_{}_lemmatized.csv'.format(i))
                        for i in range(num_cores)))
a_priori_positive_weighted = lemmatized['count_negative_weighted'].mean() / lemmatized['count_positive_weighted'].mean()
a_priori_positive = lemmatized['count_negative'].mean() / lemmatized['count_positive'].mean()
# helping my limited-memory-laptop cope with the amount of data
del reviews
del lemmatized
gc.collect()


def count_words(block_no):
    reviews = pd.read_csv('../amazon/tmp/reviews_electronics_no_text.csv')
    reviews['helpful_fraction'] = reviews['helpful'].apply(lambda x: get_fraction_difference(x))
    lemmatized = pd.read_csv('../amazon/tmp/reviews_electronics_block_{}_lemmatized.csv'.format(block_no))
    reviews = reviews.merge(lemmatized, on='reviewID', how='inner')
    del lemmatized
    
    counts_positive_negative = defaultdict(float)
    counts_positive_negative_weighted = defaultdict(float)
    counts_helpful = defaultdict(float)
    counts_rating = defaultdict(float)
    
    for index in reviews.index:
        lemmatized_text = reviews.ix[index, 'reviewTextLemmatized']
        if isinstance(lemmatized_text, str):
            split = lemmatized_text.split(' ')
            # divide by the length of the text since a single word
            # has more importance in a short text than in a long text
            word_count = len(split)
            for word in split:
                counts_positive_negative[word] += (a_priori_positive * reviews.ix[index, 'count_positive'] -
                                                   reviews.ix[index, 'count_negative']) / word_count
                counts_positive_negative_weighted[word] += \
                    (a_priori_positive_weighted * reviews.ix[index, 'count_positive_weighted'] -
                     reviews.ix[index, 'count_negative_weighted']) / word_count
                
                if reviews.ix[index, 'helpful_fraction'] > 0:
                    counts_helpful[word] += a_priori_helpful_overall / word_count
                elif reviews.ix[index, 'helpful_fraction'] < 0:
                    counts_helpful[word] -= 1 / word_count
                # leaving out 3 because that is neither really happy nor unhappy
                if reviews.ix[index, 'overall'] in {4, 5}:
                    counts_rating[word] += a_priori_good_bad_overall / word_count
                elif reviews.ix[index, 'overall'] in {1, 2}:
                    counts_rating[word] -= 1 / word_count
    all_counts = {'counts_positive_negative': counts_positive_negative,
                  'counts_positive_negative_weighted': counts_positive_negative_weighted,
                  'counts_helpful': counts_helpful,
                  'counts_rating': counts_rating}
    pickle.dump(all_counts, open('../amazon/tmp/word_counts_block_{}.pickle'.format(block_no), 'wb'))

Parallel(n_jobs=num_cores)(delayed(count_words)(i) for i in range(num_cores))
