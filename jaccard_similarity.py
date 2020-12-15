import warnings
import json
warnings.filterwarnings('ignore')
from datetime import datetime

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import pandas as pd
import numpy as np

#import texthero as hero

from heapq import nlargest
from operator import itemgetter

print("-----READING FILE--------")
meli_item  = pd.read_json('data/item_data.jl',lines=True)

#meli_item["title_clean"]=hero.clean(meli_item["title"])

def get_jaccard_sim(str1, str2):
    """
    RETURN Jaccard similarity
    """
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def parallel_similarity(domin):
    similarity = {}
    domain_title = meli_item[meli_item["domain_id"]==domin].set_index("item_id")["title"]
    items_domain = meli_item[meli_item["domain_id"]==domin]["item_id"].unique()
    for item in tqdm(items_domain):
        similarity[item]=[(i,get_jaccard_sim(domain_title[item],domain_title[i]))  for i in items_domain]
    return similarity

domain_to_search=meli_item["domain_id"].value_counts().loc[lambda x:(x>25000)].index
print("LEN OF SEARCH       :",len(domain_to_search))
print("START SIMILARITY AT :",datetime.now())
#results = Parallel(n_jobs=10)(delayed(parallel_similarity)(i) for i in meli_item["domain_id"].unique()[:9])
item_similarity_jaccard = {}
for domain in domain_to_search:
    domain_title = meli_item[meli_item["domain_id"]==domain].set_index("item_id")["title"]
    items_domain = meli_item[meli_item["domain_id"]==domain]["item_id"].unique()
    for item in tqdm(items_domain):
        full_similarity = [(i,round(get_jaccard_sim(domain_title[item],domain_title[i]),3))  for i in items_domain]
        item_similarity_jaccard[str(item)]= [nlargest(10, full_similarity, key=itemgetter(1))]

print("END SIMILARITY   AT : ",datetime.now())

pd.DataFrame.from_dict(item_similarity_jaccard,orient="index",columns=["similarity"]).to_csv("item_similarity/item_similarity_jaccard_25000_top10.csv")
