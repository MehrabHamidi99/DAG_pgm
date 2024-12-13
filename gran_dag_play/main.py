import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
import argparse
from dataset import SyntheticDataset
from torch.utils.data import DataLoader
from learnables import LearnableModel_NonLinGauss, LearnableModel_NonLinGaussANM
from train import train
from generate_data import load_graphs
from utils import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-vars', type=int, default=5)
    
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--g-path1', type=str, default='data/G1.gpickle')
    parser.add_argument('--g-path2', type=str, default='data/G2.gpickle')
    parser.add_argument('--exp_path', type=str, default='exp/')
    parser.add_argument('--output-ckpt', type=str, default='models/')

    parser.add_argument('--sem_type', type=str, default='non_linear_gaussian')
    parser.add_argument('--sigma-min', type=float, default=0.4)
    parser.add_argument('--sigma-max', type=float, default=0.8)
    parser.add_argument('--load-data', type=bool, default=True)
    parser.add_argument('--n_points', type=int, default=1000)

    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hid-dim', type=int, default=10)
    parser.add_argument('--nonlin', type=str, default='leaky-relu',
                        help="leaky-relu | sigmoid")
    
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-train-iter', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stop-crit-win', type=int, default=100,
                        help='window size to compute stopping criterion')
    parser.add_argument('--plot_freq', type=int, default=10000)
    
    parser.add_argument('--h-threshold', type=float, default=1e-8,
                        help='Stop when |h|<X. Zero means stop AL procedure only when h==0. Should use --to-dag even '
                             'with --h-threshold 0 since we might have h==0 while having cycles (due to numerical issues).')
    parser.add_argument('--mu-init', type=float, default=1e-3)
    parser.add_argument('--lambda-init', type=float, default=0)
    
    parser.add_argument('--omega-lambda', type=float, default=1e-4)
    parser.add_argument('--omega-mu', type=float, default=0.9)
    parser.add_argument('--lr-reinit', type=float, default=1e-3,
                        help='Learning rate for optim after first subproblem. Default mode reuses --lr.')
    parser.add_argument('--edge-clamp-range', type=float, default=1e-4,
                        help='as we train, clamping the edges (i,j) to zero when prod_ij is that close to zero. '
                             '0 means no clamping. Uses masks on inputs. Once an edge is clamped, no way back.')
    
    parser.add_argument('--norm-prod', type=str, default="paths",
                        help='how to normalize prod: paths|none')
    parser.add_argument('--square-prod', action="store_true",
                        help="square weights instead of absolute value in prod")
    args = parser.parse_args()
    print(args)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    G1,A1 = load_graphs(args.g_path1)
    G2,A2 = load_graphs(args.g_path2)
    gt_adjacency = get_intersection_edges(A1,A2)
    dataset_train = SyntheticDataset(args.n_points, args.g_path1, args.g_path2, args.sem_type, args.sigma_min,
                                args.sigma_max, args.load_data, args.data_path,device=args.device,mode='train')
    dataset_test = SyntheticDataset(args.n_points, args.g_path1, args.g_path2, args.sem_type, args.sigma_min,
                                args.sigma_max, args.load_data, args.data_path,device=args.device,mode='test')
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,generator=torch.Generator(device='cuda'))
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,generator=torch.Generator(device='cuda'))

    # creating the model
    model = LearnableModel_NonLinGaussANM(args.num_vars, args.num_layers, args.hid_dim, args.nonlin,args.norm_prod,args.square_prod)
    model.to(args.device)
    # learning loop
    train(model,gt_adjacency,train_loader,test_loader,args)

    

if __name__ == '__main__':
    main()