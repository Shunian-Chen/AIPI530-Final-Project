# AIPI530 Final Project

## 1. Description

This project modifies reinforcement learning repository d3rlpy to perform the following tasks

- Train CQL
  - Plot Average reward vs training epoch
    - Using existing API *evaluate_on_environment* in d3rlpy as the value of average reward
  - Compare the difference between Q Estimate and True Q
    - Using existing API *initial_state_value_estimation_scorer* in d3rlpy as the value of Q Estimate
    - Write an scorer API *true_q_value_scorer* to calculate True Q, the formula is given by $Q_{\pi_\theta}(s, a)=r(s_t, a_t) + \sum^{i=N}_{i=1}{\gamma^i*Q_{\pi_{\theta}}(s_{t+i}, a_{t+i})}$, where $Q_{\pi_\theta}(s, a)$ is the total expected reward given current state $s_t$ and action $a_t$, $r(s_t, a_t)$ is the reward of given current state $s_t$ and action $a_t$ and $\gamma$ is the discount factor

- Train OPE (FQE) to evaluate the trained policy
  - Plot Estimate Q vs Training steps
  - Using existing model *FQE* and existing API *initial_state_value_estimation_scorer*  as value Estimate Q
  
-  Compare the performance between SAC, DDPG and CQL

## 2. Codes

In the project_codes files, there are 3 python files and a Jupiter Notebook demo.

### Model

This file build model according to the given parameters, then train and save the model. 

The argument list is shown

- --dataset: a string specifying the offline dataset of desired environment, default using hopper-bullet-mixed-v0
- --seed: an integer specifying the random seed, default 0
- --q-func: a string specifying Q function used, default using mean, choices are mean, qr, iqn and fqf
- --model: a string specifying model to use, default using cql, choices are cql, ddpg, sac
- --gpu: an integer specifying whether using gpu to accelerate training process
- --save_path: a string specifying where to save the trained model and the parameters of the model
- --epoch: an integer specifying number of epochs to be performed, default 100

### OPE

This file load the existing model and evaluate its offline performance using FQE.

The argument list is primarily the same as previous step, except the load_path specifying the path to load the trained model and parameters. In general however, it should be the same as save path before.

### plot

Plot the results. There will be 3 plots for each model, which are average rewards, Q estimate vs True Q, Q estimate for OPE.

### AIPI530_Final_Project

This is a demo shows how the codes are supposed to be used. The demo can be directly run on Google Colab. However, it could also be run on local machine with some minor modification on path.

## 3. Run the code

1. Open cmd and direct to working directory

   ```shell
   cd working_dir
   ```

2. Clone the repository

   ```shell
   git clone https://github.com/Shunian-Chen/AIPI530-Final-Project
   ```

3. Install the repository

   ```shell
   pip install Cpython numpy
   python setup.py install
   ```

4. Install the environment required

   ```shell
   pip install git+https://github.com/takuseno/d4rl-pybullet
   ```

5. Train the model, passing the desired arguments. 

   ```shell
   python project_codes/Model.py --save_path sp --q-func qf --model model --epoch epoch
   ```

6. Evaluate the trained model

   ```shell
   python project_codes/OPE.py --load_path lp --q-func qr --model model --epoch epoch
   ```

7. Plot the results

   ```shell
   python project_codes/plot.py --path log_path --model model
   ```

## 4. Example Results

![CQL](https://s2.loli.net/2021/12/07/uhRWm5U2BNol7sz.png)

![DDPG](https://s2.loli.net/2021/12/07/ZwXndeECSDhNulA.png)

![SAC](https://s2.loli.net/2021/12/07/RsWQNgd8m7LOGnI.png)

## 5. Citation

```
{authors:
- family-names: "Seno"
  given-names: "Takuma"
title: "d3rlpy: An offline deep reinforcement learning library"
version: 0.91
date-released: 2020-08-01
url: "https://github.com/takuseno/d3rlpy"
preferred-citation:
  type: conference-paper
  authors:
  - family-names: "Seno"
    given-names: "Takuma"
  - family-names: "Imai"
    given-names: "Michita"
  journal: "NeurIPS 2021 Offline Reinforcement Learning Workshop"
  conference:
    name: "NeurIPS 2021 Offline Reinforcement Learning Workshop"
  collection-title: "35th Conference on Neural Information Processing Systems, Offline Reinforcement Learning Workshop, 2021"
  month: 12
  title: "d3rlpy: An Offline Deep Reinforcement Learning Library"
  year: 2021
}
> https://github.com/takuseno/d3rlpy.git 

```

