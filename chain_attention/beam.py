import torch

class Beam(object):
    def __init__(self, beam_size, device):
        self.m_b_size = beam_size
        self.m_done = False

        self.m_device = device
        self.m_scores = torch.zeros(self.m_b_size).to(self.m_device)

        self.m_prev_beam_ids = []
        self.m_next_attr_ids = []
        # self.m_next_attr_ids = [torch.LongTensor(self.m_b_size, 1).fill_(0).to(self.m_device)]

    def get_cur_state(self):
        return self.m_next_attr_ids
    
    def get_cur_origin(self):
        return self.m_prev_beam_ids[-1]

    def advance(self, preds_beam):
        voc_size = preds_beam.size(1)

        if len(self.m_prev_beam_ids) > 0:
            beam_score = preds_beam+self.m_scores.unsqueeze(1).expand_as(preds_beam)
        else:
            beam_score = preds_beam[0]
        
        flat_beam_score = beam_score.view(-1)
        # print("flat_beam_score", flat_beam_score.size())
        best_scores, best_attr_ids = flat_beam_score.topk(self.m_b_size)
        
        self.m_scores = best_scores
        # print("scores", best_scores.size())
        prev_beam_ids = best_attr_ids//voc_size
        # print("best_attr_ids", best_attr_ids)
        # print("prev_beam_ids", prev_beam_ids)
        # print("prev_beam_ids", prev_beam_ids.size())
        self.m_prev_beam_ids.append(prev_beam_ids)
        
        attr_index = best_attr_ids-prev_beam_ids*voc_size
        attr_index = attr_index.unsqueeze(-1)
        # print("attr_index", attr_index)
        if len(self.m_prev_beam_ids) == 1:
            self.m_next_attr_ids = attr_index
        else:
            tmp = self.m_next_attr_ids[prev_beam_ids]
            # print("tmp", tmp)
            # print("tmp size", tmp.size())
            self.m_next_attr_ids = torch.cat([tmp, attr_index], dim=1)
        # exit()
        return

    def sort_best(self):
        return torch.sort(self.m_scores, 0, True)
    
    def get_best(self):
        scores, ids = self.sort_best()

        return scores[1], ids[1]
    
    def get_hyp(self, k):
        hyp = self.m_next_attr_ids[k]
        # print("hyp", hyp)
        return hyp

        




