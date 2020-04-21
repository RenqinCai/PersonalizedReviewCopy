import numpy as np
from collections import Counter 

###add softmax

class _Item():
    def __init__(self):
        self.m_item_id = -1
        self.m_review_id_list = []
        self.m_valid_review_id_list = []
        self.m_word_df_map = {}
        self.m_word_tf_map = Counter()
        self.m_avg_review_words = {}
        self.m_avg_len = 0
        self.m_doc_num = 0
        self.m_total_doc_len = 0 

    def f_get_item_lm(self):

        D = len(self.m_review_id_list)
        self.m_doc_num = D
        
        self.m_avg_len = self.m_total_doc_len/D
        
        for word in self.m_word_tf_map:
            word_tf = self.m_word_tf_map[word]

            word_lm = word_tf/self.m_total_doc_len
            
            self.m_avg_review_words[word] = word_lm

    def f_get_RRe(self, review_obj, k1 = 1.5, b = 0.75):
        # for review_id in self.m_review_id_list:
        #     review_obj = review_corpus[review_id]

        # word_tf_review_map = {}
        # for word in review_obj.m_review_words:
        #     if word in stop_words:
        #         continue

        #     if word not in word_tf_review_map:
        #         word_tf_review_map[word] = 0.0
            
        #     word_tf_review_map[word] += 1.0
        
        total_word_BM25 = 0.0

        for word in review_obj.m_word_tf_map:

            word_tf = review_obj.m_word_tf_map[word]

            if word not in self.m_word_df_map:
                word_df = 0
            else:
                word_df = self.m_word_df_map[word]

            word_BM25 = word_tf*(k1+1)/(word_tf+k1*(1-b+b*self.m_doc_num/self.m_avg_len))

            word_BM25 *= np.log((self.m_doc_num-word_df+0.5)/(word_df+0.5))

            # word_tf_review_map[word] = np.exp(word_BM25)
            # total_word_BM25 += word_tf_review_map[word]

            review_obj.m_res_review_words[word] = np.exp(word_BM25)
            total_word_BM25 += review_obj.m_res_review_words[word]

        for word in review_obj.m_res_review_words:
            word_BM25 = review_obj.m_res_review_words[word]
            review_obj.m_res_review_words[word] = word_BM25/total_word_BM25

        # for word in word_tf_review_map:
        #     word_BM25 = word_tf_review_map[word]
        #     word_tf_review_map[word] = word_BM25/total_word_BM25
        
    def f_set_item_id(self, item_id):
        self.m_item_id = item_id

    def f_add_valid_review_id(self, review_id):
        self.m_valid_review_id_list.append(review_id)

    def f_add_review_id(self, review_obj, review_id):
        self.m_review_id_list.append(review_id)

        # self.m_total_doc_len += review_obj.m_word_num
        # self.m_word_tf_map = Counter(review_obj.m_word_tf_map)+self.m_word_tf_map
        # # word_review_list = []
        # # for word in word_ids:
        # #     if word in stop_words:
        # #         continue

        # #     if word not in self.m_word_tf_map:
        # #         self.m_word_tf_map[word] = 0

        # #     self.m_word_tf_map[word] += 1.0

        # #     if word not in word_review_list:
        # #         word_review_list.append(word)
            
        # #     self.m_total_doc_len += 1

        # for word in self.m_word_tf_map:
        #     if word not in self.m_word_df_map:
        #         self.m_word_df_map[word] = 0.0
            
        #     self.m_word_df_map[word] += 1.0
    
