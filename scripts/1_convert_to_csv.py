import os
import gzip
import pandas as pd

num_blocks = int(os.environ['ML_NUM_CORES'])


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df(path):
    df = {}
    for i, d in enumerate(parse(path)):
        df[i] = d
    return pd.DataFrame.from_dict(df, orient='index')

df = get_df('../amazon/reviews_Electronics_5.json.gz')
df.reset_index(inplace=True)
df.rename(columns={'index': 'reviewID'}, inplace=True)

for dir_ in ['../amazon/tmp/', '../amazon/output/']:
    try:
        os.mkdir(dir_)
    except:
        pass

# save into blocks for text processing
block_size = int(len(df) / num_blocks) + 1
for i in range(num_blocks):
    df.loc[i * block_size: min(len(df) - 1, (i + 1) * block_size), ['reviewID', 'reviewText']].to_csv(
        '../amazon/tmp/reviews_electronics_block_{}.csv'.format(i), index=False, header=True)

# save everything apart from the text into separate file
del df['reviewText']
df.to_csv('../amazon/tmp/reviews_electronics_no_text.csv', index=False, header=True)
