import argparse
import d3rlpy
import torch as th
import os
from d3rlpy.algos import CQL
from d3rlpy.datasets import get_pybullet
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import true_q_value_scorer
from sklearn.model_selection import train_test_split
from d3rlpy.gpu import Device
from d3rlpy.models.optimizers import AdamFactory


def main(args):
    dataset, env = get_pybullet(args.dataset)

    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    device = None if args.gpu is None else Device(args.gpu)

    optim_factory = AdamFactory(weight_decay=1e-4)

    cql = CQL(  q_func_factory=args.q_func, 
                use_gpu=device, 
                scaler = "standard", 
                action_scaler='min_max',
                optim_factory=optim_factory)

    save_path = os.path.curdir if args.save_path == None else args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    para_path = os.path.join(save_path, "params.json")
    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=100,
            scorers={
                'average_reward': evaluate_on_environment(env),
                'Q_estimate': initial_state_value_estimation_scorer,
                "True_Q": true_q_value_scorer
            },
            path = para_path)


    
    save_path = os.path.join(save_path, "model.pt")
    cql.save_model(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='hopper-bullet-mixed-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--q-func',
                        type=str,
                        default='mean',
                        choices=['mean', 'qr', 'iqn', 'fqf'])
    gpu = 0 if th.cuda.is_available() else None
    parser.add_argument('--gpu', type=int, default = gpu)
    parser.add_argument('--save_path', type=str, default = None)
    args = parser.parse_args()
    main(args)