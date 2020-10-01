import torch

def cov_diag(m):
    before = m.shape[1:]
    m = m.view(m.size(0), -1)
    factor = 1.0 / (m.size(0) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    diag_cov = factor * torch.einsum("ij,ij->j", (m, m))
    return diag_cov.view(before)

class ControllerSettings(object):
    def __init__(
        self, 
        search_space,
        max_len=20,
        hidden_size=488,
        embedding_size=256,
        learning_rate = 1e-5,
        device = 'cpu'
    ):
        self.search_space = search_space
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.device = device