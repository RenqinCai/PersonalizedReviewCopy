import sys
import os
import math
from collections import Counter
import multiprocessing
from multiprocessing import Pool
import argparse
import stanza
import pickle
import datetime
import pandas as pd
import math

# nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos')
# exit()
GPUS = [0, 1, 2, 3]

# def f_extract_feature_wrapper(args):
#     return f_extract_feature(args)

def f_extract_feature(args):

    id, review_list_process = args
    gpu_id = str(GPUS[id % len(GPUS)])
    print("gpu id", gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos')

    counts = Counter()
    total_review_process = len(review_list_process)
    # nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos', use_gpu=False)
    print("review num process", total_review_process)
    for idx, review in enumerate(review_list_process):
        # print("idx", idx)
        doc = nlp(review)
    
        features = []
        for sen in doc.sentences:
            words = sen.words
            for word in words:
                if word.xpos in {'NN', 'NNS', 'NNP', 'NNPS'}:
                    word_text = word.text
                    if word_text not in features:
                        features.append(word_text)

        features = set(features)        
        counts.update(features)
    
        if idx % 1e4 == 0:
            print("idx", idx)
    print("process end")
    return counts

def main():
    pickle_file = "/p/reviewde/data/yelp/yelp_filter_20_restaurant.pickle"
    df = pd.read_pickle(pickle_file)

    review_list = df.review.tolist()
    review_num = len(review_list)
    print("review num", review_num)
    # exit()
    start_time = datetime.datetime.now()

    worker_num = 24
    with Pool(worker_num) as p:
        per_worker = math.ceil(review_num/worker_num)
        # params = [(i, per_worker*i, per_worker*(i+1)) for i in range(worker_num)]
        # for i in range(worker_num):
        #     print("per worker", per_worker*i, per_worker*(i+1))

        params = [(i, review_list[per_worker*i: per_worker*(i+1)]) for i in range(worker_num)]

        results = p.map(f_extract_feature, params)

    total_features = sum(results, Counter())
    total_features = sorted(total_features.items(), key=lambda x: x[1], reverse=True)

    feature_file = "feature_yelp_restaurant.txt"

    f = open(feature_file, "w")

    f.write('\n'.join( f'{f}, {c}' for f, c in total_features))

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("duration", duration)

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    main()


