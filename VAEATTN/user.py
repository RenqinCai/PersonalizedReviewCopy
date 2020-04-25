class _User():
    def __init__(self):
        self.m_user_id = -1
        self.m_review_id_list = []
        self.m_word_freq_map = {}

    def f_get_user_lm(self):
        self.m_user_lm = {}

    def f_set_user_id(self, user_id):
        self.m_user_id = user_id
    
    def f_add_review_id(self, review_id):
        self.m_review_id_list.append(review_id)