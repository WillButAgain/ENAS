import torch
from NASmodels import NASController, MasterModel, EvolutionController
from utils import ControllerSettings
from tqdm import tqdm
import mlflow

# only for plotting
import numpy as np
import matplotlib.pyplot as plt

cont_settings = ControllerSettings(mask_ops=True, learning_rate=1e-2, device='cuda:0', search_space = MasterModel(keys=[], mask_ops=True).search_space, max_len=20, hidden_size=488, embedding_size=256)

n_models = 800
k=40
iterations = 1000

EA = EvolutionController(cont_settings, n_models).to(cont_settings.device)
EA.initialize_models()

scores = []
iterable = tqdm(range(iterations))

# https://docs.python.org/2/library/profile.html#module-cProfile
# mlflow.tracking.set_tracking_uri('file:/share/lazy/will/ConstrastiveLoss/Logs')
mlflow.tracking.set_tracking_uri('file:/share/lazy/will/ConstrastiveLoss/Molecules/mlruns')
mlflow.set_experiment('Evolutionary autoencoder NAS')

run_name = 'First genetic algorithm iteration'
with mlflow.start_run(run_name = run_name) as run:

    import cProfile
    # p = cProfile.run('''
    # for _ in tqdm(range(iterations)):
    #     score = EA(k)
    #     scores.append(score)
    #     iterable.set_description('Score: {}'.format(score))
    # ''', 'restats')

    for _ in tqdm(range(iterations)):
        score = EA(k)
        scores.append(score)
        iterable.set_description('Score: {}'.format(score.item()))
        mlflow.log_metric('Score', score.item(), step=_)
        mlflow.log_metric('Cov mean', EA.state_dict()['hidden_to_embedding_weight_cov_matrix'].mean().item(), step=_)
