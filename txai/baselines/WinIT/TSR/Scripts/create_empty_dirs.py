import argparse
import pathlib


def create_empty_dirs(args):
    """Creates required folder structure to run benchmark and MNIST experiments"""

    # TODO: Original code from the research paper is very inconsistent with argparse args, most of the directories are
    #       hardcoded strings. Could go through and fix

    required_dirs = []
    # Datasets
    required_dirs += [args.data_dir]
    # Graphs
    required_dirs += [args.Graph_dir, args.datasets_graphs_dir, args.Saliency_Maps_graphs_dir,
                      args.Graph_dir + '/Saliency_Distribution', args.Graph_dir + '/Precision_Recall',
                      args.Graph_dir + '/Accuracy_Drop']
    # Models
    required_dirs += ['Models', 'Models/Transformer', 'Models/TCN', 'Models/LSTMWithInputCellAttention',
                      'Models/LSTM']
    # Results
    required_dirs += [args.Saliency_dir, args.Mask_dir, args.Masked_Acc_dir, args.Precision_Recall_dir,
                      args.Acc_Metrics_dir, 'Results/Saliency_Distribution']
    # MNIST_Experiments
    required_dirs += ['MNIST_Experiments/Models', 'MNIST_Experiments/Graphs', 'MNIST_Experiments/Data']

    for directory in required_dirs:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # Mimic arguments from run_benchmark
    parser = argparse.ArgumentParser()

    parser.add_argument('--Graph_dir', type=str, default='Graphs/')
    parser.add_argument('--datasets_graphs_dir', type=str, default='Graphs/Datasets/')
    parser.add_argument('--Saliency_Maps_graphs_dir', type=str, default='Graphs/Saliency_Maps/')

    parser.add_argument('--data_dir', type=str, default="Datasets/")
    parser.add_argument('--Saliency_dir', type=str, default='Results/Saliency_Values/')
    parser.add_argument('--Mask_dir', type=str, default='Results/Saliency_Masks/')
    parser.add_argument('--Masked_Acc_dir', type=str, default="Results/Masked_Accuracy/")
    parser.add_argument('--Precision_Recall_dir', type=str, default='Results/Precision_Recall/')
    parser.add_argument('--Acc_Metrics_dir', type=str, default='Results/Accuracy_Metrics/')

    create_empty_dirs(parser.parse_args())
