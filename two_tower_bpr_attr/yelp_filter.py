import sys
import os
import json
import re
import math
import time
import multiprocessing
from multiprocessing import Pool
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import datetime
import csv

GPUS = [1, 2, 3]

def parse(sentences, features):

    exps = []
    exp_flag = False
    for sen in sentences:
        words = word_tokenize(sen)
        for word in words:
            if word in features:
                exp_flag = True
                break

        if exp_flag:
            exps.append(sen)
            exp_flag = False
        # else:
        #     print("remove", sen)

    return exps

def clean_sentence(s):
    s = re.sub(r'https?://.*', ' ', s)
    s = re.sub(r'\.\.+', '...', s)
    s = re.sub(r'`', '\'', s)
    s = re.sub(r'^\s*-', '', s)\
        .replace('*', ' ')\
        .replace('-', ' - ')\
        .replace('/', ' / ')\
        .replace('~', ' ')

    s = re.sub(r'\s\s+', ' ', s).strip().lower()

    return s

def extract(args):
    gpu_id, features, userid_list_process, itemid_list_process, review_list_process, rating_list_process = args
    print("gpu id", gpu_id)

    # gpu_id = str(GPUS[gpu_id%len(GPUS)])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    new_userid_list_process = []
    new_itemid_list_process = []
    new_review_list_process = []
    new_rating_list_process = []

    review_num_process = len(review_list_process)
    print("review num process", review_num_process)

    for idx, review in enumerate(review_list_process):
        review = review.replace('\n', ' ')
        sentences = sent_tokenize(review)
        sentences = [clean_sentence(s) for sen in sentences for s in sen.split("\n")]
        sentences = [s for s in sentences if s and len(s) < 400 and len(s) > 3]

        exps = parse(sentences, features)
        if len(exps) == 0:
            continue

        exps = " ".join(exps)

        userid = userid_list_process[idx]
        itemid = itemid_list_process[idx]
        rating = rating_list_process[idx]

        new_userid_list_process.append(userid)
        new_itemid_list_process.append(itemid)
        new_review_list_process.append(exps)
        new_rating_list_process.append(rating)

    return new_userid_list_process, new_itemid_list_process, new_review_list_process, new_rating_list_process

def main():

    feature_file = "../data/yelp_restaurant_new/attr.csv"
    f = open(feature_file, "r")
    # reader = csv.reader(f)
    # features = list(reader) 
    # feature_num = len(features)
    features = []
    for line in f:
        feature_i = line.strip()
        features.append(feature_i)
    f.close()
    feature_num = len(features)
    print("feautre num", feature_num)

    # print(features)

    pickle_file = "/p/reviewde/data/yelp_restaurant_new/yelp.pickle"
    df = pd.read_pickle(pickle_file)

    # test_num = 100000

    # review_list = df.review.tolist()[:test_num]
    # userid_list = df.userid.tolist()[:test_num]
    # itemid_list = df.itemid.tolist()[:test_num]
    
    review_list = df.review.tolist()
    userid_list = df.userid.tolist()
    itemid_list = df.itemid.tolist()
    rating_list = df.rating.tolist()

    review_num = len(review_list)
    print("review num", review_num)
    

    start_time = datetime.datetime.now()

    n_workers = 24

    with Pool(n_workers) as p:
        per_worker = math.ceil(review_num/n_workers)
        params = [(i, features, userid_list[per_worker*i: per_worker*(i+1)], itemid_list[per_worker*i: per_worker*(i+1)], review_list[per_worker*i: per_worker*(i+1)], rating_list[per_worker*i: per_worker*(i+1)]) for i in range(n_workers)]
        
        results = p.map(extract, params)

    userid_list = []
    itemid_list = []
    review_list = []
    rating_list = []

    for result in results:
        userid_list_process = result[0]
        itemid_list_process = result[1]
        review_list_process = result[2]
        rating_list_process = result[3]

        userid_list.extend(userid_list_process)
        itemid_list.extend(itemid_list_process)
        review_list.extend(review_list_process)
        rating_list.extend(rating_list_process)

    user_num = len(userid_list)
    item_num = len(itemid_list)
    review_num = len(review_list)
    print("user num:%d, item num:%d, review num:%d"%(user_num, item_num, review_num))

    data_list = []
    for idx in range(user_num):
        userid = userid_list[idx]
        itemid = itemid_list[idx]
        review = review_list[idx]
        rating = rating_list[idx]

        data_i = [userid, itemid, review, rating]
        data_list.append(data_i)

    new_df = pd.DataFrame(data_list)
    new_df.columns = ['userid', 'itemid', 'review', 'rating']
    print("columns", new_df.columns)
    yelp_file = "../data/yelp_restaurant_new/yelp_restaurant.pickle"
    new_df.to_pickle(yelp_file)

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("duration", duration)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
    
