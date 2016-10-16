from collections import defaultdict
from joblib import Parallel, delayed
import os
import pandas as pd
from spacy.en import English

num_cores = int(os.environ['ML_NUM_CORES'])

posneg = pd.read_csv('../ext/posneg.csv', encoding='ISO-8859-1')
posneg = {row['word']: row['strength'] for index, row in posneg.iterrows()}
negations = {'no', 'noone', 'none', 'neither', 'not', 'zilch',
             'nobody', 'nothing', 'nought', 'none', 'never'}


def check_for_negations(word):
    # checks how many negations refer to this word
    sum_ = sum([check_for_negations(child) for child in word.children])
    if word.lemma_ in negations:
        sum_ += 1
    return sum_


def compute_block(block_no):
    reviews = pd.read_csv('../amazon/tmp/reviews_electronics_block_{}.csv'.format(block_no))
    parser = English()
    result = []
    for idx in reviews.index:
        if isinstance(reviews.ix[idx, 'reviewText'], str):
            parsed_review = parser(reviews.ix[idx, 'reviewText'])
            counts = defaultdict(int)
            for word in parsed_review:
                if word.lemma_ in posneg.keys():
                    current_word = word
                    # find the head of all heads
                    while current_word.head is not current_word:
                        current_word = current_word.head
                    n_negations = check_for_negations(current_word)

                    weight = posneg[word.lemma_]
                    if n_negations % 2:
                        # means the statement was negated
                        weight *= -1
                    if weight < 0:
                        counts['count_negative'] += 1
                        counts['count_negative_weighted'] -= weight
                    else:
                        counts['count_positive'] += 1
                        counts['count_positive_weighted'] += weight
                if word.lemma_ in negations:
                    counts['count_negations'] += 1
            # this is a little uglier than writing to a dataframe immediately but _a lot_ faster
            result.append([reviews.ix[idx, 'reviewID'],
                           " ".join([x.lemma_ for x in parsed_review]),
                           counts['count_negations'],
                           counts['count_negative'],
                           counts['count_negative_weighted'],
                           counts['count_positive'],
                           counts['count_positive_weighted']])
        else:
            result.append([reviews.ix[idx, 'reviewID'], None, None, None, None, None, None])

    pd.DataFrame(result, columns=['reviewID', 'reviewTextLemmatized', 'count_negations', 'count_negative',
                                  'count_negative_weighted', 'count_positive', 'count_positive_weighted']).to_csv(
        '../amazon/tmp/reviews_electronics_block_{}_lemmatized.csv'.format(block_no))

Parallel(n_jobs=num_cores)(delayed(compute_block)(i) for i in range(num_cores))
