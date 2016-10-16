import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

num_blocks = int(os.environ['ML_NUM_CORES'])

reviews = pd.concat((pd.read_csv('../amazon/tmp/reviews_electronics_block_{}_lemmatized.csv'.format(i))
                    for i in range(num_blocks)))

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=300) 
count_matrix = vectorizer.fit_transform((reviews.ix[idx, 'reviewTextLemmatized']
                                         if isinstance(reviews.ix[idx, 'reviewTextLemmatized'], str) else ''
                                         for idx in reviews.index))
                                                      
with open('../amazon/output/tfidf_vocabulary.json', 'w') as outfile:
    json.dump(vectorizer.vocabulary_, outfile)

tfidf = TfidfTransformer()
tfidf_matrix = tfidf.fit_transform(count_matrix, reviews.ix[:, 'overall'])
tfidf_matrix = pd.DataFrame(tfidf_matrix.todense())
vocabulary = {value: key for key, value in vectorizer.vocabulary_.items()}
tfidf_matrix.columns = [vocabulary[col] for col in tfidf_matrix.columns]
tfidf_matrix['reviewID'] = reviews['reviewID'].values
tfidf_matrix.to_csv('../amazon/tmp/tfidf_matrix.csv', index=False, header=True)
