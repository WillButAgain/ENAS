import torch
import numpy as np
from torch import nn 

def cov(m, rowvar=True):
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def corrcoef(m):
    c = cov(m)
    try:
        d = torch.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = torch.sqrt(d)
    c /= stddev[:, None]
    c /= stddev[None, :]
    return c

def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob.detach()


def eval_score(jacob):
    corrs = corrcoef(jacob.squeeze())
    # [:, 1] because torch.eig returns complex eigenvalues too for whatever fucking reason
    v = torch.eig(corrs)[0][:, 0]
    k = 1e-5
    ret = -torch.mean(torch.log(v + k) + 1./(v + k))
    return ret if (ret != np.inf) else -150

class StdConv(nn.Module):
    def __init__(self, C_in, C_out):
        super(StdConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PoolBranch(nn.Module):
    def __init__(self, pool_type, C_in, C_out, kernel_size, stride, affine=False):
        super().__init__()
        self.preproc = StdConv(C_in, C_out)
        self.pool = Pool(pool_type, kernel_size, stride, padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        out = self.preproc(x)
        out = self.pool(out)
        out = self.bn(out)
        return out

class UpsampleBranch(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, affine=False):
        super().__init__()
        self.preproc = StdConv(C_in, C_out)
        self.convtranspose = nn.ConvTranspose1d(C_out, C_out, kernel_size, stride)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        out = self.preproc(x)
        out = self.pool(out)
        out = self.bn(out)
        return out
    
class SeparableConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(C_in, C_in, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride,
                                   groups=C_in, bias=False)
        self.pointwise = nn.Conv1d(C_in, C_out, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBranch(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, separable):
        super().__init__()
        self.preproc = StdConv(C_in, C_out)
        if separable:
            self.conv = SeparableConv(C_out, C_out, kernel_size, stride)
        else:
            self.conv = nn.Conv1d(C_out, C_out, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.postproc = nn.Sequential(
            nn.BatchNorm1d(C_out, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.preproc(x)
        out = self.conv(out)
        out = self.postproc(out)
        return out

class ResidualConvBranch(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, separable):
        super().__init__()
        self.preproc = StdConv(C_in, C_out)
        if separable:
            self.conv = SeparableConv(C_out, C_out, kernel_size, stride)
        else:
            self.conv = nn.Conv1d(C_out, C_out, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.postproc = nn.Sequential(
            nn.BatchNorm1d(C_out, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.preproc(x)
        out = self.conv(out)
        out = self.postproc(out)
        return out+x

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = nn.Conv1d(C_in, C_out, 1, stride=2, padding=0, bias=False)
#         self.conv2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
#         out = torch.cat([self.conv1(x), self.conv2(x[:, 1:])], dim=1)
        out = self.conv1(x)
        out = self.bn(out)
        return out

class FactorizedUpsample(nn.Module):
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(C_in, C_out, 1, stride=2, padding=0, bias=False)
#         self.conv2 = nn.ConvTranspose1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
#         out = torch.cat([self.conv1(x), self.conv2(x[:, 1:])], dim=1)
        out = self.conv1(x)
        out = self.bn(out)
        return out
    
class Pool(nn.Module):
    def __init__(self, pool_type, kernel_size, stride, padding):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride, padding=padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride, padding=padding)
        else:
            raise ValueError()

    def forward(self, x):
        return self.pool(x)


class SepConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = SeparableConv(C_in, C_out, kernel_size, 1)
        self.bn = nn.BatchNorm1d(C_out, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
    
    
    
# utils.py

import logging
from collections import OrderedDict

import numpy as np
import torch

_counter = 0

_logger = logging.getLogger(__name__)


def global_mutable_counting():
    """
    A program level counter starting from 1.
    """
    global _counter
    _counter += 1
    return _counter


def _reset_global_mutable_counting():
    """
    Reset the global mutable counting to count from 1. Useful when defining multiple models with default keys.
    """
    global _counter
    _counter = 0


def to_device(obj, device):
    """
    Move a tensor, tuple, list, or dict onto device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_device(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def to_list(arr):
    if torch.is_tensor(arr):
        return arr.cpu().numpy().tolist()
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return arr


class AverageMeterGroup:
    """
    Average meter group for multiple average meters.
    """

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        """
        Update the meter group with a dict of metrics.
        Non-exist average meters will be automatically created.
        """
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        """
        Return a summary string of group data.
        """
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeter:
    """
    Computes and stores the average and current value.
    Parameters
    ----------
    name : str
        Name to display.
    fmt : str
        Format string to print the values.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with value and weight.
        Parameters
        ----------
        val : float or int
            The new value to be accounted in.
        n : int
            The weight of the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class StructuredMutableTreeNode:
    """
    A structured representation of a search space.
    A search space comes with a root (with `None` stored in its `mutable`), and a bunch of children in its `children`.
    This tree can be seen as a "flattened" version of the module tree. Since nested mutable entity is not supported yet,
    the following must be true: each subtree corresponds to a ``MutableScope`` and each leaf corresponds to a
    ``Mutable`` (other than ``MutableScope``).
    Parameters
    ----------
    mutable : nni.nas.pytorch.mutables.Mutable
        The mutable that current node is linked with.
    """

    def __init__(self, mutable):
        self.mutable = mutable
        self.children = []

    def add_child(self, mutable):
        """
        Add a tree node to the children list of current node.
        """
        self.children.append(StructuredMutableTreeNode(mutable))
        return self.children[-1]

    def type(self):
        """
        Return the ``type`` of mutable content.
        """
        return type(self.mutable)

    def __iter__(self):
        return self.traverse()

    def traverse(self, order="pre", deduplicate=True, memo=None):
        """
        Return a generator that generates a list of mutables in this tree.
        Parameters
        ----------
        order : str
            pre or post. If pre, current mutable is yield before children. Otherwise after.
        deduplicate : bool
            If true, mutables with the same key will not appear after the first appearance.
        memo : dict
            An auxiliary dict that memorize keys seen before, so that deduplication is possible.
        Returns
        -------
        generator of Mutable
        """
        if memo is None:
            memo = set()
        assert order in ["pre", "post"]
        if order == "pre":
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
        for child in self.children:
            for m in child.traverse(order=order, deduplicate=deduplicate, memo=memo):
                yield m
        if order == "post":
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
                    
                    
# ==========================================================================    
# Stacked LSTM cell and ENAS mutator
# ==========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutator import Mutator
from nni.nas.pytorch.mutables import LayerChoice, InputChoice, MutableScope


class StackedLSTMCell(nn.Module):
    def __init__(self, layers, size, bias):
        super().__init__()
        self.lstm_num_layers = layers
        self.lstm_modules = nn.ModuleList([nn.LSTMCell(size, size, bias=bias)
                                           for _ in range(self.lstm_num_layers)])

    def forward(self, inputs, hidden):
        prev_c, prev_h = hidden
        next_c, next_h = [], []
        for i, m in enumerate(self.lstm_modules):
            curr_c, curr_h = m(inputs, (prev_c[i], prev_h[i]))
            next_c.append(curr_c)
            next_h.append(curr_h)
            # current implementation only supports batch size equals 1,
            # but the algorithm does not necessarily have this limitation
            inputs = curr_h[-1].view(1, -1)
        return next_c, next_h


class EnasMutator(Mutator):
    """
    A mutator that mutates the graph with RL.
    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    lstm_size : int
        Controller LSTM hidden units.
    lstm_num_layers : int
        Number of layers for stacked LSTM.
    tanh_constant : float
        Logits will be equal to ``tanh_constant * tanh(logits)``. Don't use ``tanh`` if this value is ``None``.
    cell_exit_extra_step : bool
        If true, RL controller will perform an extra step at the exit of each MutableScope, dump the hidden state
        and mark it as the hidden state of this MutableScope. This is to align with the original implementation of paper.
    skip_target : float
        Target probability that skipconnect will appear.
    temperature : float
        Temperature constant that divides the logits.
    branch_bias : float
        Manual bias applied to make some operations more likely to be chosen.
        Currently this is implemented with a hardcoded match rule that aligns with original repo.
        If a mutable has a ``reduce`` in its key, all its op choices
        that contains `conv` in their typename will receive a bias of ``+self.branch_bias`` initially; while others
        receive a bias of ``-self.branch_bias``.
    entropy_reduction : str
        Can be one of ``sum`` and ``mean``. How the entropy of multi-input-choice is reduced.
    """

    def __init__(self, model, lstm_size=64, lstm_num_layers=2, tanh_constant=1.5, cell_exit_extra_step=False,
                 skip_target=0.4, temperature=None, branch_bias=0.25, entropy_reduction="sum"):
        super().__init__(model)
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.cell_exit_extra_step = cell_exit_extra_step
        self.skip_target = skip_target
        self.branch_bias = branch_bias

        self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
        self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]), requires_grad=False)  # pylint: disable=not-callable
        assert entropy_reduction in ["sum", "mean"], "Entropy reduction must be one of sum and mean."
        self.entropy_reduction = torch.sum if entropy_reduction == "sum" else torch.mean
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.bias_dict = nn.ParameterDict()

        self.max_layer_choice = 0
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                if self.max_layer_choice == 0:
                    self.max_layer_choice = len(mutable)
                assert self.max_layer_choice == len(mutable), \
                    "ENAS mutator requires all layer choice have the same number of candidates."
                # We are judging by keys and module types to add biases to layer choices. Needs refactor.
                if "reduce" in mutable.key:
                    def is_conv(choice):
                        return "conv" in str(type(choice)).lower()
                    bias = torch.tensor([self.branch_bias if is_conv(choice) else -self.branch_bias  # pylint: disable=not-callable
                                         for choice in mutable])
                    self.bias_dict[mutable.key] = nn.Parameter(bias, requires_grad=False)

        self.embedding = nn.Embedding(self.max_layer_choice + 1, self.lstm_size)
        self.soft = nn.Linear(self.lstm_size, self.max_layer_choice, bias=False)

    def sample_search(self):
        self._initialize()
        self._sample(self.mutables)
        return self._choices

    def sample_final(self):
        return self.sample_search()

    def _sample(self, tree):
        mutable = tree.mutable
        if isinstance(mutable, LayerChoice) and mutable.key not in self._choices:
            self._choices[mutable.key] = self._sample_layer_choice(mutable)
        elif isinstance(mutable, InputChoice) and mutable.key not in self._choices:
            self._choices[mutable.key] = self._sample_input_choice(mutable)
        for child in tree.children:
            self._sample(child)
        if isinstance(mutable, MutableScope) and mutable.key not in self._anchors_hid:
            if self.cell_exit_extra_step:
                self._lstm_next_step()
            self._mark_anchor(mutable.key)

    def _initialize(self):
        self._choices = dict()
        self._anchors_hid = dict()
        self._inputs = self.g_emb.data
        self._c = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self._h = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self.sample_log_prob = 0
        self.sample_entropy = 0
        self.sample_skip_penalty = 0

    def _lstm_next_step(self):
        self._c, self._h = self.lstm(self._inputs, (self._c, self._h))

    def _mark_anchor(self, key):
        self._anchors_hid[key] = self._h[-1]

    def _sample_layer_choice(self, mutable):
        self._lstm_next_step()
        logit = self.soft(self._h[-1])
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * torch.tanh(logit)
        if mutable.key in self.bias_dict:
            logit += self.bias_dict[mutable.key]
        branch_id = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
        log_prob = self.cross_entropy_loss(logit, branch_id)
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        self._inputs = self.embedding(branch_id)
        return F.one_hot(branch_id, num_classes=self.max_layer_choice).bool().view(-1)

    def _sample_input_choice(self, mutable):
        query, anchors = [], []
        for label in mutable.choose_from:
            if label not in self._anchors_hid:
                self._lstm_next_step()
                self._mark_anchor(label)  # empty loop, fill not found
            query.append(self.attn_anchor(self._anchors_hid[label]))
            anchors.append(self._anchors_hid[label])
        query = torch.cat(query, 0)
        query = torch.tanh(query + self.attn_query(self._h[-1]))
        query = self.v_attn(query)
        if self.temperature is not None:
            query /= self.temperature
        if self.tanh_constant is not None:
            query = self.tanh_constant * torch.tanh(query)

        if mutable.n_chosen is None:
            logit = torch.cat([-query, query], 1)  # pylint: disable=invalid-unary-operand-type

            skip = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip_prob = torch.sigmoid(logit)
            kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl
            log_prob = self.cross_entropy_loss(logit, skip)
            self._inputs = (torch.matmul(skip.float(), torch.cat(anchors, 0)) / (1. + torch.sum(skip))).unsqueeze(0)
        else:
            assert mutable.n_chosen == 1, "Input choice must select exactly one or any in ENAS."
            logit = query.view(1, -1)
            index = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip = F.one_hot(index, num_classes=mutable.n_candidates).view(-1)
            log_prob = self.cross_entropy_loss(logit, index)
            self._inputs = anchors[index.item()]

        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        return skip.bool()
    
    
    
    
def load_full_state(model_to_update, optimizer_to_update, Path, freeze_weights=False):
    """
    Load the model and optimizer state_dict, and the total number of epochs
    The use case for this is if we care about the optimizer state_dict, which we do if we have multiple training 
    sessions with momentum and/or learning rate decay. this will track the decay/momentum.

    Args: 
            model_to_update (Module): Pytorch model with randomly initialized weights. These weights will be updated.
            optimizer_to_update (Module): Optimizer with your learning rate set. 
            THIS FUNCTION WILL NOT UPDATE THE LEARNING RATE YOU SPECIFY.
            Path (string): If we are not training from scratch, this path should be the path to the "run_stats" file in the artifacts 
            directory of whatever run you are using as a baseline. 
            You can find the path in the MLFlow UI. It should end in /artifacts/run_stats  
            

    Returns:
            Nothing

    Note:
            The model and optimizer will not be returned, rather the optimizer and module you pass to this function will be modified.
    """
    checkpoint = torch.load(Path)
    
    # freeze weights of the first model
    update_dict = {k: v for k, v in checkpoint['model'].items() if k in model_to_update.state_dict()}
                # do this so it does not use the learning rate from the previous run. this is unwanted behavior
                # in our scenario since we are not using a learning rate scheduler, rather we want to tune the learning
                # rate further after we have gotten past the stalling
            #     checkpoint['optimizer']['param_groups'][0]['lr'] = optimizer_to_update.state_dict()['param_groups'][0]['lr']
            #     optimizer_to_update.load_state_dict(checkpoint['optimizer'])
    
    # to go back to old behavior, just do checkpoint['model'] instead of update_dict
    model_to_update.load_state_dict(update_dict, strict=False)
    
    print('Of the '+str(len(model_to_update.state_dict())/2)+' parameter layers to update in the current model, '+str(len(update_dict)/2)+' were loaded')