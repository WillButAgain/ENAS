import torch.nn as nn


def Embedding(num_embeddings, embedding_dim):
    """
    Simple wrapper for embedding layers that handles initialization
    Args:
        num_embeddings: vocab size
        embedding_dim: embedding dimension
        padding_idx: padding index
    Returns:
        torch.nn.Embedding properly initialized
    """
    m = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
#         padding_idx=padding_idx,
    )
#     self.std = (embedding_dim ** -0.5)
#     self.std = 1.
    nn.init.normal_(m.weight, mean=0, std=1)
#     nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=False):
    """
    Simple wrapper for Linear layers that handles initialization
    Args:
        in_features: input dimension
        out_features: output dimension
        bias: bool, whether or not to add bias
    Returns:
        torch.nn.Linear properly initialized
    """
    m = nn.Linear(in_features, out_features, bias)
    nn.init.normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0)
    return m

def GRUCell(input_size: int, hidden_size: int, bias: bool = False):
    """
    Simple wrapper for GRUCell that handles initialization
    Args:
        input_size – The number of expected features in the input x
        hidden_size – The number of features in the hidden state h
        bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    Returns:
        h’ of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
    """
    m = nn.GRUCell(input_size, hidden_size, bias)

#     self.stdv = 1.0 / math.sqrt(hidden_size)
#     nn.init.uniform_(m.weight_ih, -self.stdv, self.stdv)
#     nn.init.uniform_(m.weight_hh, -self.stdv, self.stdv)

    nn.init.normal_(m.weight_ih)
    nn.init.normal_(m.weight_hh)

    if bias:
        nn.init.constant_(m.bias_ih, 0)
        nn.init.constant_(m.bias_hh, 0)

    return m