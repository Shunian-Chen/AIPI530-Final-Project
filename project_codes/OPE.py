import os
import argparse
import torch as th
from d3rlpy.algos import CQL
from d3rlpy.datasets import get_pybullet
from d3rlpy.ope import FQE
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer


def main(args):
    load_path = args.load_path
    para_path = os.path.join(load_path, "params.json")
    model_path = os.path.join(load_path, "model.pt")

    # prepare the trained algorithm
    cql = CQL.from_json(para_path)
    cql.load_model(model_path)

    # dataset to evaluate with
    dataset, env = get_pybullet('hopper-bullet-mixed-v0')

    # off-policy evaluation algorithm
    fqe = FQE(algo=cql)
    
    # train estimators to evaluate the trained policy
    fqe.fit(dataset.episodes,
        eval_episodes=dataset.episodes,
        n_epochs=100,
        scorers={
           'init_value': initial_state_value_estimation_scorer,
        })

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
    parser.add_argument('--load_path', type=str, default = None)
    args = parser.parse_args()
    main(args)