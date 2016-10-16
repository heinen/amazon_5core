from collections import defaultdict
import json
import numpy
import os
import pandas as pd
import pickle

num_blocks = int(os.environ['ML_NUM_CORES'])


def get_fraction(x):
    x = x[1:-1].split(',')
    if int(x[1]) == 0:
        return numpy.nan
    return float(x[0]) / float(x[1])

# bringing it all together
lemmatized = pd.concat((pd.read_csv('../amazon/tmp/reviews_electronics_block_{}_lemmatized.csv'.format(i))
                        for i in range(num_blocks)))
reviews = pd.read_csv('../amazon/tmp/reviews_electronics_no_text.csv')
reviews = reviews.merge(lemmatized, on='reviewID', how='inner')
reviews['helpful_fraction'] = reviews['helpful'].apply(lambda x: get_fraction(x))
reviews['helpful_number_of_ratings'] = reviews['helpful'].apply(lambda x: x[1:-1].split(',')[1])

word_count_blocks = [pickle.load(open('../amazon/tmp/word_counts_block_{}.pickle'.format(block_no), 'rb'))
                     for block_no in range(num_blocks)]
word_counts_combined = defaultdict(lambda: defaultdict(int))
for word_count_dict in word_count_blocks:
    for count_type, count_dict in word_count_dict.items():
        for key, value in count_dict.items():
            word_counts_combined[count_type][key] += value
            
relevant_words = {}
for key, value in word_counts_combined.items():
    tmp = sorted(value, key=value.get, reverse=True)
    # keep it as list because json can't handle sets
    relevant_words[key + '_bad'] = tmp[-200:]
    relevant_words[key + '_good'] = tmp[:200]
    
# I discovered these by accident on a subset of the full data set
# when computing the relevant words
relevant_words['spanish'] = ['uso', 'pantalla', 'gran',
                             'hace', 'compra', 'permite', 'recomiendo', 'producto',
                             'serios', 'bien', 'excelente', 'panos', 'tu', 'contrarion',
                             'tiene', 'calidad', 'esta', 'buena', 'mejor', 'este', 'como',
                             'pero', 'por', 'para', 'muy', 'una', 'que']
    
with open('../amazon/output/relevant_words.json', 'w') as outfile:
    json.dump(relevant_words, outfile)
    
# transform it into sets for feature counting
for key in relevant_words.keys():
    relevant_words[key] = set(relevant_words[key])

# write away once, spares dataframe overhead
all_texts = reviews['reviewTextLemmatized'].values
# now actually count which category occurred how many times in which review
for relevant_words_category, relevant_words_set in relevant_words.items():
    column = []
    for i, idx in enumerate(reviews.index):
        if isinstance(all_texts[i], str):
            column.append(sum((1 for word in all_texts[i].split(' ')
                               if word in relevant_words_set)))
        else:
            column.append(numpy.nan)
    reviews[relevant_words_category + '_count'] = column

reviews['count_words'] = [len(x.split(' ')) if isinstance(x, str) else 0 for x in all_texts]
del reviews['reviewTextLemmatized']
reviews.to_csv('../amazon/tmp/review_features.csv', index=False, header=True)
