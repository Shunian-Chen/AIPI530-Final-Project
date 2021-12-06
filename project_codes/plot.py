import argparse
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

def main(args):
    path = args.path
    model = args.model.upper()
    train_scores = ["Q_estimate", "True_Q", "average_reward"]
    ope_scores = ['init_value']
    file_path = {}
    for score in train_scores:
        file_path[score] = (join(path, model, score + '.csv'))
    for score in ope_scores:
        file_path[score] = (join(path, 'FQE_{}'.format(model), score + '.csv'))

    files = {}
    for file, f_path in file_path.items():
        files[file] = pd.read_csv(f_path, header=None)
    
    fig, ax = plt.subplots(len(file_path)-1, 1, figsize = (60, 60))
    fig.suptitle(model)

    
    #plot average reward
    axis = ax[0]
    axis.plot(files["average_reward"].iloc[:, 2])
    axis.set_title("Average Reward vs Training Epoch")

    #plot Q_estimate and True_Q
    axis = ax[1]
    qe, = axis.plot(files["Q_estimate"].iloc[:, 2])
    tq, = axis.plot(files["True_Q"].iloc[:, 2])
    axis.set_title("Q Estimate vs True Q")
    axis.legend(handles = [qe, tq], labels = ["Q Estimate", "True Q"], loc = "upper left")

    #plot OPE
    axis = ax[2]
    axis.plot(files["init_value"].iloc[:, 2])
    axis.set_title("OPE Estimate Q vs Training Epoch")

    epoch = files['average_reward'].shape[0]
    plt.savefig('Results/{}_{}.png'.format(model, epoch))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, default = None)
    parser.add_argument('--model',
                        type=str,
                        default='cql',
                        choices=['cql', 'ddpg', 'sac'])
    args = parser.parse_args()
    main(args)