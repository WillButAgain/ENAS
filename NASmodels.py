from torch import nn
import torch
import numpy as np
from layers import Linear, GRUCell, Embedding
from utils import cov_diag

def vec_translate(a, Dict):    
    return np.vectorize(Dict.__getitem__)(a)

class MasterModel(nn.Module):
    
    def __init__(self, keys, n=32):
        super().__init__() 
        self.input_keys = keys
#         self.args = args

        
        self.UpOrDown = nn.ModuleDict({
            'Up':nn.ConvTranspose1d(n, n, kernel_size=2, stride=2),
            'Down':nn.MaxPool1d(n)
        })
        
        self.search_space = nn.ModuleDict({
            '<SOS>':nn.Identity(),
            'conv3':nn.Conv1d(n, n, kernel_size=3, stride=1, padding=(3-1)//2),
            'conv5':nn.Conv1d(n, n, kernel_size=5, stride=1, padding=(5-1)//2),
            'conv7':nn.Conv1d(n, n, kernel_size=7, stride=1, padding=(7-1)//2),
            'conv9':nn.Conv1d(n, n, kernel_size=9, stride=1, padding=(9-1)//2),
            'conv11':nn.Conv1d(n, n, kernel_size=11, stride=1, padding=(11-1)//2),
            'conv13':nn.Conv1d(n, n, kernel_size=13, stride=1, padding=(13-1)//2),
            'conv15':nn.Conv1d(n, n, kernel_size=15, stride=1, padding=(15-1)//2),
            'BN':nn.BatchNorm1d(n, n),
            'Relu':nn.ReLU(),
            'LeakyRelu':nn.LeakyReLU(0.01),
            
            # Experimental
            'Switch':nn.Identity(),
            'UpOrDown':self.UpOrDown,

            # Should always be at the bottom
            '<EOS>':nn.Identity(),
        })
        self.n_keys = len(self.search_space.keys())+2
        
        # DO NOT FORGET - you can reference modules in ModuleList by index! 
        # this is how you will do skip connections later, me!
        self.model = nn.ModuleList()
        
        # we need this defined so it saves to the state_dict
        self.initial_conv = nn.Conv1d(1, n, kernel_size=15, padding=(15-1)//2)
        self.model.append(self.initial_conv)
        
        dim_change_op_list = []
        dim_change_op = 'Down'
        for i, key in enumerate(keys):
            if key == 'UpOrDown':
                dim_change_op_list.append(dim_change_op)
                self.model.append(self.search_space[key][dim_change_op])
                continue
                
            if key == 'Switch':
                dim_change_op = 'Up'
                continue
                
            self.model.append(self.search_space[key])        

        self.model = nn.Sequential(*self.model)
        
        # this is going to be the reward mechanism to make sure it doesnt make non-valid autoencoders - AND, if its zero, we know to continue training
        self.good_model = 0
        for op in dim_change_op_list:
            if op == 'Down':
                self.good_model += 1
            if op == 'Up':
                self.good_model -= 1

        self.dim_change_ops = 0
        for op in dim_change_op_list:
            if (op == 'Down') | (op == 'Up'):
                self.dim_change_ops += 1

    # DELETE - i do not think this will ever be used
    def define_search_space(self, search_space):
        self.search_space = search_space
        
    def save_weights(self):
        self.state_dict()['search_space.initial_conv.weight'] = self.state_dict()['initial_conv.weight']
        self.state_dict()['search_space.initial_conv.bias'] = self.state_dict()['initial_conv.bias']
        
        for i, key in enumerate(self.input_keys):
            if 'conv' in key:
                self.state_dict()['search_space.' + key + '.weight'+'.'+str(i+1)] = self.state_dict()['model.' + str(i+1) + '.weight']
                self.state_dict()['search_space.' + key + '.bias'+'.'+str(i+1)] = self.state_dict()['model.' + str(i+1) + '.bias']
                print('Updated {} weights'.format('search_space.' + key + '.weight'+'.'+str(i+1)))

#         save_dict = {}
#         for key in self.state_dict().keys():
#             if 'search_space' in key:
#                 save_dict[key] = self.state_dict[key]
        # fuck it 
        torch.save(self.state_dict, '/share/lazy/will/ML/RLDatabase/MasterWeights.pyt')

        
    def load_weights(self):
        master_update = torch.load('/share/lazy/will/ML/RLDatabase/MasterWeights.pyt')
        
        self.state_dict()['initial_conv.weight'] = master_update['search_space.initial_conv.weight']
        self.state_dict()['initial_conv.bias'] = master_update['search_space.initial_conv.bias']
        
        for i, key in enumerate(self.input_keys):
            if 'conv' in key:
                self.state_dict()['model.' + str(i+1) + '.weight'] = master_update['search_space.' + key + '.weight'+'.'+str(i+1)]
                self.state_dict()['model.' + str(i+1) + '.bias'] = master_update['search_space.' + key + '.bias'+'.'+str(i+1)]
                print('Updated {} weights'.format('search_space.' + key + '.weight'+'.'+str(i+1)))

    def forward(self, x):
        return self.model(x)
        
        
class NASController(nn.Module):
    def __init__(self, controller_settings):
        super().__init__()
        self.hidden_size = controller_settings.hidden_size
        self.embedding_size = controller_settings.embedding_size
        self.max_len = controller_settings.max_len
        self.search_space = controller_settings.search_space
        self.n_keys = len(controller_settings.search_space.keys())
        
        embedding_size = 256
        self.hidden_to_embedding = Linear(self.hidden_size, self.n_keys)
        
        self.GRU = GRUCell(
            input_size = self.embedding_size,
            hidden_size = self.hidden_size,
#             bias=False,
        )
        
        self.embed = Embedding(
            num_embeddings = self.n_keys, 
            embedding_dim = self.embedding_size, 
        )

        # Initialize hidden states for forward()
        self.embedding = torch.zeros((1, self.embedding_size))
        self.hidden_state = torch.zeros((1, self.hidden_size))

        self.tokenizer_dict = dict((key, i) for i, key in enumerate(self.search_space.keys()))

    def forward(self):
        embedding_list = []
        for _ in range(self.max_len):
            # propagate hidden state
            self.hidden_state = self.GRU(self.embedding, self.hidden_state)
            predicted_embedding = self.hidden_to_embedding(self.hidden_state)
            self.embedding = self.embed(torch.max(predicted_embedding, dim=-1)[1])
            embed_index = torch.max(predicted_embedding, dim=-1)[1].item()
            if embed_index == self.n_keys-1:
                break
            embedding_list.append(embed_index)
            
        model_tokens = self.get_tokens(embedding_list)
        return MasterModel(keys=model_tokens)
        
    

    def tokenize(self, input_keys):
        return [vec_translate(token, self.tokenizer_dict).tolist() for token in input_keys]

    def get_tokens(self, input_indices):
        self.get_tokens_dict = dict((v, k) for k, v in self.tokenizer_dict.items())
        return [vec_translate(token, self.get_tokens_dict).tolist() for token in input_indices]
        

    
from operator import itemgetter 
class EvolutionController(nn.Module):
    def __init__(self, controller_settings, n_models):
        super().__init__()
        self.controller_settings = controller_settings
        self.n_models = n_models
        
        # Setup initial controller to grab relevant params and shapes for initialization
        # should only done once at init
        self.controller_params = []
        self.cov_matrices = []
        self.means_matrices = []
        init_state_dict = NASController(controller_settings).state_dict()
        for param_group in init_state_dict.keys():
            # ignore search space
            if 'search_space' not in param_group:
                self.controller_params.append(param_group)
                # Buffers cannot contain periods. Why? No idea...
                self.means_matrices.append(param_group.replace('.', '_') + '_means_matrix')
                self.cov_matrices.append(param_group.replace('.', '_') + '_cov_matrix')
        
        # Since we do not need a separate covariance/means matrix for each model, we instead register them as a buffer in the global scope
        # We do this in hopes that it will save to the state_dict, but also in fear that it will double-count ModuleList state_dicts
        for param_group, cov_matrix_name, means_matrix_name in zip(self.controller_params, self.cov_matrices, self.means_matrices):
            self.register_buffer(cov_matrix_name, torch.ones_like(init_state_dict[param_group]))
            self.register_buffer(means_matrix_name, torch.zeros_like(init_state_dict[param_group]))
                            
        del init_state_dict


    def initialize_models(self):        
        # initialize update dict that will hold hidden state updates 
        self.update_dict_list = [{} for _ in range(self.n_models)]
        
        # initialize models - cant think of better way to store than in ModuleList 
        # We call no_grad here because we shouldnt anywhere else - we will need grad later, but not here
#         with torch.no_grad():
        self.model_list = nn.ModuleList([NASController(self.controller_settings) for _ in range(self.n_models)])
        
        self.evolve_()

    def update_cov_matrix_(self, model_list):
        for param_group, mean_tensor, cov_tensor in zip(self.controller_params, self.cov_matrices, self.means_matrices):
            # Concat the parameter tensors at dim=0
            param_tensor = torch.cat([model.state_dict()[param_group].unsqueeze(0) for model in model_list], dim=0)

            # Estimate the diagonal of the covariance matrix, and the mean
            # WARNING: when the number of selected models becomes large, i worry about numerical stability
            unbiased_cov_estimate = cov_diag(param_tensor)
            mean_estimates = param_tensor.mean(dim=0)
            
            # I am missing an embarassing amount of intricacies in the covariance matrix update - way way way off
#             step_coef = step_size(n_chosen=5, population_size=25)
            step_coef = self.controller_settings.learning_rate
            
            self.state_dict()[cov_tensor] += step_coef*unbiased_cov_estimate
            self.state_dict()[mean_tensor] += step_coef*mean_estimates
        
        
    def evolve_(self):
        '''
        Pytorch convention is to end any function name with _ if it is an inplace operation
        
        Notice that this does not take a model_list argument whereas update_cov_matrix does - every population member
        will get an update here, whereas when we compute the cov matrix, we are only looking at a subset of the 
        population
        '''
        update_dict_list = [{} for _ in range(self.n_models)]
        for param_group, mean_tensor, cov_tensor in zip(self.controller_params, self.means_matrices, self.cov_matrices):
            dist = torch.distributions.normal.Normal(
                loc=self.state_dict()[mean_tensor], 
                scale=self.state_dict()[cov_tensor]
            )
            weight_update = dist.sample((self.n_models,))
            
            # accumulate the update dictionaries
            for i in range(self.n_models):
                update_dict_list[i][param_group] = weight_update[i]
            self.update_dict = self.state_dict()[mean_tensor]
        for model, update_dict in zip(self.model_list, update_dict_list):
#             self.update_dict = update_dict
            model.load_state_dict(update_dict, strict=False)


    def forward(self, k):
        scores = torch.zeros(self.n_models, 2)
        # by arbitrary choice, we choose to maximize score
        for i, model in enumerate(self.model_list):
            scores[i, 0] = 10 - model().good_model
            scores[i, 1] = model().dim_change_ops
        
        survived_indices = torch.topk(scores.sum(axis=1), k=4).indices
        
        self.update_cov_matrix_(itemgetter(*survived_indices)(self.model_list))

        self.evolve_()
        print(scores.sum())
            