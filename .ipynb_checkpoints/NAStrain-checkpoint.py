import mlflow
import logging
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim

from nni.nas.pytorch import mutables
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup, to_device

# my thing
from model.NASutils import get_batch_jacobian, eval_score,  FactorizedReduce, ConvBranch, PoolBranch, FactorizedUpsample, EnasMutator, ResidualConvBranch

logger = logging.getLogger(__name__)



class ENASLayer(mutables.MutableScope):
    def __init__(self, key, prev_labels, in_filters, out_filters):
        super().__init__(key)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.mutable = mutables.LayerChoice([
#             ConvBranch(in_filters, out_filters, kernel_size=3, stride=1, separable=False),
#             ConvBranch(in_filters, out_filters, kernel_size=3, stride=1, separable=True),
            ConvBranch(in_filters, out_filters, kernel_size=5, stride=1, separable=False),
#             ConvBranch(in_filters, out_filters, kernel_size=5, stride=1, separable=True),
#             ConvBranch(in_filters, out_filters, kernel_size=7, stride=1, separable=False),
#             ConvBranch(in_filters, out_filters, kernel_size=7, stride=1, separable=True),
#             ConvBranch(in_filters, out_filters, kernel_size=9, stride=1, separable=False),
            ConvBranch(in_filters, out_filters, kernel_size=41, stride=1, separable=True),
#             ResidualConvBranch(in_filters, out_filters, kernel_size=3, stride=1, separable=False),
#             ResidualConvBranch(in_filters, out_filters, kernel_size=3, stride=1, separable=True),
#             ResidualConvBranch(in_filters, out_filters, kernel_size=5, stride=1, separable=False),
#             ResidualConvBranch(in_filters, out_filters, kernel_size=5, stride=1, separable=True),
#             ResidualConvBranch(in_filters, out_filters, kernel_size=7, stride=1, separable=False),
#             ResidualConvBranch(in_filters, out_filters, kernel_size=7, stride=1, separable=True),
#             ResidualConvBranch(in_filters, out_filters, kernel_size=9, stride=1, separable=False),
#             ResidualConvBranch(in_filters, out_filters, kernel_size=9, stride=1, separable=True),
#             PoolBranch('avg', in_filters, out_filters, kernel_size=3, stride=1),
#             PoolBranch('max', in_filters, out_filters, kernel_size=3, stride=1),
        ])
        if len(prev_labels) > 0:
            self.skipconnect = mutables.InputChoice(choose_from=prev_labels, n_chosen=None)
        else:
            self.skipconnect = None
        self.batch_norm = nn.BatchNorm1d(out_filters, affine=False)

    def forward(self, prev_layers):
        out = self.mutable(prev_layers[-1])
        if self.skipconnect is not None:
            connection = self.skipconnect(prev_layers[:-1])
            if connection is not None:
                out += connection
        return self.batch_norm(out)


class GeneralNetwork(nn.Module):
    def __init__(self, num_layers=12, out_filters=24, in_channels=1,
                 dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.out_filters = out_filters

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_filters, 25, 1, 1, bias=False),
            nn.BatchNorm1d(out_filters)
        )
        # when to downsample
        pool_distance = self.num_layers // 6
        self.pool_layers_idx = [pool_distance - 1, 2 * pool_distance - 1]
        # when to upsample
        upsample_distance = self.num_layers // 6
        self.upsample_layers_idx = [3 * pool_distance - 1, 4 * pool_distance - 1]

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        labels = []
        for layer_id in range(self.num_layers):
            labels.append("layer_{}".format(layer_id))
            
            if layer_id in self.pool_layers_idx:
#                 self.pool_layers.append(FactorizedReduce(self.out_filters, self.out_filters))
                self.pool_layers.append(nn.MaxPool1d(2))
            
            if layer_id in self.upsample_layers_idx:
                self.upsample_layers.append(FactorizedUpsample(self.out_filters, self.out_filters))
                
            self.layers.append(ENASLayer(labels[-1], labels[:-1], self.out_filters, self.out_filters))            


    def forward(self, x):
        bs = x.size(0)
        cur = self.stem(x)

        layers = [cur]

        for layer_id in range(self.num_layers):
            cur = self.layers[layer_id](layers)
            layers.append(cur)
            if layer_id in self.pool_layers_idx:
                for i, layer in enumerate(layers):
                    layers[i] = self.pool_layers[self.pool_layers_idx.index(layer_id)](layer)
                cur = layers[-1]

#         cur = self.gap(cur).view(bs, -1)
#         cur = self.dropout(cur)
#         logits = self.dense(cur)
        return cur #logits
    
    
    
class EnasTrainer(Trainer):
    '''
     model, metrics, 
                 optimizer, num_epochs, dataloader_train, dataloader_valid,
                 mutator=None, batch_size=64, device=None, log_frequency=None, callbacks=None,
                 entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999, child_steps=500,
                 mutator_lr=0.00035, mutator_steps_aggregate=20, mutator_steps=50, aux_weight=0.4,
                 test_arc_per_epoch=1
     '''
#     self.previous_lr = 0.00035
    def __init__(self, model, mutator, 
                 optimizer, num_epochs, dataloader_train, dataloader_valid,
                 batch_size=64, device=None, log_frequency=None, callbacks=None,
                 entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999, child_steps=500,
                 mutator_lr=0.00035, mutator_steps_aggregate=20, mutator_steps=50, aux_weight=0.4,
                 test_arc_per_epoch=1):
        super().__init__(model, mutator, nn.CrossEntropyLoss(), None, optimizer, num_epochs,
                 dataloader_train, dataloader_valid, batch_size, None, device, log_frequency, callbacks)
        self.model = model.to(device)
        self.log_frequency = log_frequency
        self.optimizer = optimizer
        self.device = device
        self.callbacks = callbacks
        self.num_epochs = num_epochs
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.mutator = mutator
        self.mutator_optim = optim.Adam(self.mutator.parameters(), lr=mutator_lr)
        self.batch_size = batch_size
#         self.metrics = metrics
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.
        self.mutator_steps_aggregate = mutator_steps_aggregate
        self.mutator_steps = mutator_steps
        self.child_steps = child_steps
        self.aux_weight = aux_weight
        self.test_arc_per_epoch = test_arc_per_epoch

        self.init_dataloader()

    def init_dataloader(self):
        self.train_loader = cycle(self.dataloader_train)
        self.valid_loader = cycle(self.dataloader_valid)

    def train_one_epoch(self, epoch):
        # Train sampler (mutator)
        self.model.eval()
        self.mutator.train()
        total_loss=0
        meters = AverageMeterGroup()
        for mutator_step in range(1, self.mutator_steps + 1):
            self.mutator_optim.zero_grad()
            for step in range(1, self.mutator_steps_aggregate + 1):
                x, y = next(self.valid_loader)
                x, y = to_device(x, self.device), to_device(y, self.device)

                self.mutator.reset()
                with torch.no_grad():
                    logits = self.model(x)
                self._write_graph_status()
                jacobian = get_batch_jacobian(self.model, x)
                jacobian = jacobian.reshape(jacobian.size(0), -1)

                reward = eval_score(jacobian)
                total_loss += reward.item()
            
                if self.entropy_weight:
                    reward += self.entropy_weight * self.mutator.sample_entropy.item()
                # https://arxiv.org/pdf/1707.06347.pdf
                self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
                loss = self.mutator.sample_log_prob * (reward - self.baseline)
                if self.skip_weight:
                    loss += self.skip_weight * self.mutator.sample_skip_penalty

                loss /= self.mutator_steps_aggregate
                loss.backward()

                cur_step = step + (mutator_step - 1) * self.mutator_steps_aggregate
                if self.log_frequency is not None and cur_step % self.log_frequency == 0:
                    logger.info("RL Epoch [%d/%d] Step [%d/%d] [%d/%d]  %s", epoch + 1, self.num_epochs,
                                mutator_step, self.mutator_steps, step, self.mutator_steps_aggregate,
                                reward.int())

            nn.utils.clip_grad_norm_(self.mutator.parameters(), 5.)
            self.mutator_optim.step()
        mlflow.log_metric('Total reward', -total_loss/(self.mutator_steps*self.mutator_steps_aggregate), epoch)
        torch.save({
        'model':self.mutator.state_dict(),
        'optimizer':self.mutator_optim.state_dict()
        }, 'mutator_run_stats.pyt')
        mlflow.log_artifact('mutator_run_stats.pyt')

    def validate_one_epoch(self, epoch):
        pass

