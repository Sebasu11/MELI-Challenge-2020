import warnings
import json
warnings.filterwarnings('ignore')
from datetime import datetime

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import pandas as pd
import numpy as np

from collections import Counter
from operator import itemgetter
from heapq import nlargest
get_recent = itemgetter("event_timestamp")

import re
import unicodedata

def clean_text(text) :
    text=unicodedata.normalize('NFKD', str(text)).encode('ascii', errors='ignore').decode('utf-8')\
    .lower().replace(r'\\n','').replace(r'-',' ').replace(r'.','').strip()
    text = re.sub("[^a-zA-Z]", " ", str(text))
    return " ".join(text.split())

def get_count_search(row):
    log=[]
    for event in row:
        if event["event_type"]=="search":
            log.append(event["event_info"])
    if len(log)==0: return "NONE"
    else: return dict(Counter(log))


def get_jaccard_sim(str1, str2):
    """
    RETURN Jaccard similarity
    """
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def find_search_domain_simil(txt):
    jacc_sim = {}
    for k in domain_dict.keys():
        jacc_sim[k]=get_jaccard_sim(txt,domain_dict[k])
    return jacc_sim

def get_count_search(row):
    log=[]
    for event in row:
        if event["event_type"]=="search":
            log.append(event["event_info"])
    if len(log)==0: return "NONE"
    else: return dict(Counter(log))

print("-----READING FILE--------")
meli_item  = pd.read_json('data/item_data.jl',lines=True)
train = pd.read_json('data/train_dataset.jl', lines=True)


train["count_search"]=train["user_history"].apply(get_count_search)

train["search_text"]=train["count_search"].apply(get_text_search)

search_domain_df = train[["search_text","domain_id"]].dropna()

domain_dict=meli_item.groupby(["domain_id"])["item_id"].count().to_dict()
for k in domain_dict.keys():
    domain_dict[k]=clean_text(k[3:])


print("START SIMILARITY AT : ",datetime.now())
results = Parallel(n_jobs=10)(delayed(find_search_domain_simil)(i) for i in search_domain_df["search_text"].values[:9])
print("END SIMILARITY   AT : ",datetime.now())

simil_item ={}
for res in results:
    k = list(res.keys())[0]
    simil_item[k]=res[k]

with open('item_similarity_jaccard.json', 'w') as fh:
    json.dump(simil_item, fh)
