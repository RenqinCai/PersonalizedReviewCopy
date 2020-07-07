import numpy as np
from collections import Counter 

class _Review():
    def __init__(self):
        self.m_review_id = -1
        self.m_review_words = []
        self.m_res_review_words = {}
        self.m_avg_review_words = {}
        self.m_user_id = -1
        self.m_item_id = -1
        self.m_word_tf_map = {}
        self.m_informative_word_num = 0

        self.m_user_perturb_words = []
        self.m_item_perturb_words = []
        self.m_local_perturb_words = []
        
    # def f_set_review(self, review_id, review_words, word_tf_map, informative_word_num):
    #     self.m_review_id = review_id
    #     self.m_review_words = review_words

    #     self.m_word_tf_map = word_tf_map

    #     self.m_informative_word_num = informative_word_num
    
    def f_set_review(self, review_id, review_words):
        self.m_review_id = review_id
        self.m_review_words = review_words

    def f_set_pertub_review(self, new_review_user, new_review_item, new_review_local):
        self.m_user_perturb_words = new_review_user
        self.m_item_perturb_words = new_review_item
        self.m_local_perturb_words = new_review_local

    def f_set_user_item(self, user_id, item_id):
        self.m_user_id = user_id
        self.m_item_id = item_id