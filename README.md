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



## 2. Installation

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

   The argument list is shown

   - --dataset: a string specifying the offline dataset of desired environment, default using hopper-bullet-mixed-v0
   - --seed: an integer specifying the random seed, default 0
   - --q-func: a string specifying Q function used, default using mean, choices are mean, qr, iqn and fqf
   - --model: a string specifying model to use, default using cql, choices are cql, ddpg, sac
   - --gpu: an integer specifying whether using gpu to accelerate training process
   - --save_path: a string specifying where to save the trained model and the parameters of the model
   - --epoch: an integer specifying number of epochs to be performed, default 100

6. Evaluate the trained model

   ```shell
   python "project_codes/OPE.py" --load_path lp --q-func qr --model model --epoch epoch
   ```

   The argument list is primarily the same as previous step, except the load_path specifying the path to load the trained model and parameters. In general however, it should be the same as save path before.

7. Plot the results

