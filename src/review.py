import numpy as np
from collections import Counter 

class _Review():
    def __init__(self):
        self.m_review_id = -1
        self.m_review_words = []
        self.m_res_review_words = {}
        self.m_user_id = -1
        self.m_item_id = -1
        self.m_word_tf_map = {}
        self.m_word_num = 0
        
    def f_set_review(self, review_id, review_words, stop_words):
        self.m_review_id = review_id
        self.m_review_words = review_words

        word_tf_map = Counter(review_words)

        for word in word_tf_map:
            if word in stop_words:
                continue

            self.m_word_tf_map[word] = word_tf_map[word]
            
        self.m_word_num = sum(self.m_word_tf_map.values())

    def f_set_user_item(self, user_id, item_id):
        self.m_user_id = user_id
        self.m_item_id = item_id