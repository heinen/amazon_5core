from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # so that matplotlib does not try to show us windows with the latest figure
import matplotlib.pyplot as plt
import pandas as pd
import pickle

reviews = pd.read_csv('../amazon/tmp/review_features.csv')
reviews['reviewTime'] = reviews['unixReviewTime'].apply(lambda x: (x - 929232000)/(1406073600-929232000) * 
                                                        (2014.47671232 - 1999.45205479) + 1999.45205479)

print('Average overall score is {}'.format(reviews['overall'].mean()))
figsize = (14, 8)
plt.figure(1)
reviews['overall'].hist(range=[1, 5], bins=9, figsize=figsize)
plt.savefig('../amazon/output/overall_rating_distribution.png')
plt.figure(2)
reviews[reviews['spanish_count'] > 2]['helpful_fraction'].hist(figsize=figsize)
plt.savefig('../amazon/output/helpful_fraction_for_spanish_ge2.png')
plt.figure(3)
reviews['counts_helpful_good_count'].hist(range=[0, 150], bins=150, figsize=figsize)
plt.savefig('../amazon/output/counts_helpful_good_count.png')
plt.figure(4)
reviews['counts_helpful_bad_count'].hist(range=[0, 400], bins=400, figsize=figsize)
plt.savefig('../amazon/output/counts_helpful_bad_count.png')
plt.figure(5)
reviews['counts_rating_good_count'].hist(range=[0, 200], bins=200, figsize=figsize)
plt.savefig('../amazon/output/counts_rating_good_count.png')
plt.figure(6)
reviews['counts_rating_bad_count'].hist(range=[0, 200], bins=200, figsize=figsize)
plt.savefig('../amazon/output/counts_rating_bad_count.png')
plt.figure(7)
(reviews['count_positive'] - reviews['count_negative']).hist(range=[-30, 30], bins=60, figsize=figsize)
plt.savefig('../amazon/output/difference_count_positive_count_negative.png')
plt.figure(8)
reviews['helpful_fraction'].hist(range=[0, 1], bins=10, figsize=figsize)
plt.savefig('../amazon/output/helpful_fraction.png')
plt.figure(9)
reviews['count_positive_weighted'].hist(range=[0, 70], bins=70, figsize=figsize)
plt.savefig('../amazon/output/count_positive_weighted.png')
plt.figure(10)
reviews['count_positive'].hist(range=[0, 60], bins=60, figsize=figsize)
plt.savefig('../amazon/output/count_positive.png')
plt.figure(11)
reviews['count_negative_weighted'].hist(range=[0, 60], bins=60, figsize=figsize)
plt.savefig('../amazon/output/count_negative_weighted.png')
plt.figure(12)
reviews['count_negative'].hist(range=[0, 35], bins=35, figsize=figsize)
plt.savefig('../amazon/output/count_negative.png')
plt.figure(13)
reviews['reviewTime'].hist(figsize=figsize, bins=100, range=[1999, 2015])
plt.savefig('../amazon/output/reviewTime.png')
plt.figure(14)
(reviews['count_negative']/reviews['count_words']).hist(range=[0, 1], bins=60, figsize=figsize)
plt.savefig('../amazon/output/rel_count_negative.png')
plt.figure(15)
(reviews['count_positive']/reviews['count_words']).hist(range=[0, 1], bins=60, figsize=figsize)
plt.savefig('../amazon/output/rel_count_positive.png')

one_month = 0.083333
first_review_time = reviews['reviewTime'].min()
number_of_months = 1 + int((reviews['reviewTime'].max() - first_review_time) / one_month)
rating_by_month = [[] for month in range(number_of_months)]
for idx in reviews.index:
    rating_by_month[int((reviews.ix[idx, 'reviewTime'] - 
                         first_review_time) / one_month)].append(reviews.ix[idx, 'overall'])
tmp1 = [first_review_time + month * one_month for month in range(number_of_months)]
tmp2 = [(sum(x) / len(x)) for x in rating_by_month]
plt.figure(num=16, figsize=figsize)
plt.plot(pd.Series(tmp1[10:]), pd.Series(tmp2[10:]))
plt.savefig('../amazon/output/rating_by_month.png')

# hist showing how the weights are useless
plt.figure(17)
posneg = pd.read_csv('../ext/posneg.csv', encoding='ISO-8859-1')
posneg['strength'].hist(figsize=figsize)
plt.savefig('../amazon/output/pos_neg_weights.png')


# proof of concept part                                                                               
def ddd():
    # only defined to be able to unpickle
    return defaultdict(dict) 
    
reviewer_dict = pickle.load(open('../amazon/tmp/reviewer_dict.pickle', 'rb'))
# -------------------------------- Isolate reviewers with too many good ratings --------------------------------
for reviewer_id, product_reviews in reviewer_dict.items():
    # I tested this with much lower thresholds and the results were still
    # very usable but when turned to 60, only one reviewer remains and 
    # he's my favorite (and definitely rewarded for his reviews)
    if len(product_reviews) < 60:
        continue
    if all((review['overall'] == 5 for review_id, review in product_reviews.items())):
        print('The following reviewer has nothing but 5* ratings, these are the according reviewIDs:')
        print(reviewer_id)
        for review_id, review in product_reviews.items():
            print(review_id)
        print('---------------------------')
# ---------------------------------------------------------------------------------------------------------------
        
number_of_reviews = [len(product_reviews) for reviewer_id, product_reviews in reviewer_dict.items()]
plt.figure(18)
pd.Series(number_of_reviews).hist(range=[0, 35], bins=35, figsize=figsize)
plt.savefig('../amazon/output/number_of_reviews_per_reviewer.png')


class OuterBreak(Exception):
    pass

# -------------------------------- Isolate reviewers that are simply not helpful --------------------------------
print('The following reviewerIDs were identified to be frauds only by language:')
for reviewer_id, product_reviews in reviewer_dict.items():
    tmp = []
    # These restrictions are very, very tight (no helpful
    # ratings at all, basically nothing belonging to the helpful word 
    # list) to showcase the biggest frauds only by language. 
    # When taking away the no-helpful-rating restriction and slightly
    # loosening the restriction on how many 'helpful words' are allowed,
    # the result set is shockingly large and (after a quick manual check)
    # it still seems to be quite accurate
    try:
        for review_id, review in product_reviews.items():
            if review['helpful_number_of_ratings'] == 0 and review['overall'] >= 4:
                tmp.append(review['counts_helpful_good_count'] / review['counts_helpful_bad_count'])
            else:
                raise OuterBreak
        if len(tmp) > 5 and sum(tmp)/len(tmp) <= 0.2:
                    print(reviewer_id)
    except OuterBreak:
        pass
print('------------------------------')
# ---------------------------------------------------------------------------------------------------------------
del reviewer_dict

product_dict = pickle.load(open('../amazon/tmp/product_dict.pickle', 'rb'))
time_window = 0.0833333333333333 * 2  # two months
# -------------------------------- Isolate products with sudden positive spikes --------------------------------
print('The following productIDs have sudden spikes of positive ratings over time:')
for product_id, product_reviews in product_dict.items():
    try:
        # this is a little heuristic because it is not known
        # how long the product was online for
        review_times = [value['reviewTime'] for key, value in product_reviews.items()]
        number_of_months = (max(review_times) - min(review_times)) / time_window
        ratings_per_month = len(product_reviews) / number_of_months

        ratings_over_time = sorted((product_review['reviewTime'], product_review['overall'],
                                    product_review['counts_helpful_good_count'],
                                    product_review['counts_helpful_bad_count'])
                                   for key, product_review in product_reviews.items())
        windows = []
        for i in range(len(ratings_over_time)):
            window = []
            for j in range(i, len(ratings_over_time)):
                if ratings_over_time[i][0] + time_window < ratings_over_time[j][0]:
                    break
                window.append(ratings_over_time[j])
            if (len(window) > 1.5 * ratings_per_month and len(window) > 20 and
                sum((x[1] for x in window)) / len(window) > 4 and
                    sum((x[2] / x[3] for x in window)) / len(window) < 0.5):
                windows.append(window)
        if len(windows) > 1:
            print(product_id)
    except KeyError:
        pass
print('------------------------------')
# ---------------------------------------------------------------------------------------------------------------

print('The following reviews were identified as useless by language:')
for idx in reviews.index:
    if (reviews.ix[idx, 'count_words'] > 10 and
            reviews.ix[idx, 'counts_helpful_bad_count'] / reviews.ix[idx, 'count_words'] == 1):
        print(reviews.ix[idx, 'reviewID'])
