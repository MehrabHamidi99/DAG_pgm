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
from train import train,pns,to_dag
from generate_data import load_graphs
from utils import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-vars', type=int, default=5)
    parser.add_argument('--train', action="store_true",
                        help='Run `train` function, get /train folder')
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--g-path1', type=str, default='data/G1.gpickle')
    parser.add_argument('--g-path2', type=str, default='data/G2.gpickle')
    parser.add_argument('--g-path3', type=str, default=None)
    parser.add_argument('--exp_path', type=str, default='exp/')
    parser.add_argument('--output-ckpt', type=str, default='models/')

    parser.add_argument('--sem_type', type=str, default='non_linear_gaussian')
    parser.add_argument('--sigma-min', type=float, default=0.4)
    parser.add_argument('--sigma-max', type=float, default=0.8)
    parser.add_argument('--load-data', type=bool, default=True)
    parser.add_argument('--n_points', type=int, default=10000)

    parser.add_argument('--num-neighbors', type=int, default=None,
                        help='number of neighbors to select in PNS')
    parser.add_argument('--pns-thresh', type=float, default=0.75,
                        help='threshold in PNS')
    parser.add_argument('--pns', action="store_true",
                        help='Run `pns` function, get /pns folder')
    
    parser.add_argument('--to-dag', action="store_true",
                        help='Run `to-dag` function, get /to-dag folder')
    parser.add_argument('--jac_thresh', action="store_true",help='threshold using the Jacobian instead of prod')


    parser.add_argument('--cam-pruning', action="store_true",
                        help='Run `cam_pruning` function, get /cam-pruning folder')
    parser.add_argument('--cam-pruning-cutoff', nargs='+',
                        default=np.logspace(-6, 0, 10),
                        help='list of cutoff values. Higher means more edges')
    
    # Risk extrapolation hyperparms
    parser.add_argument('--risk-extrapolation', action="store_true",
                        help='Do risk extrapolation')
    parser.add_argument('--beta', type=float, default=0.5,help='Regularizer intensity')

    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hid-dim', type=int, default=10)
    parser.add_argument('--nonlin', type=str, default='leaky-relu',
                        help="leaky-relu | sigmoid")
    
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-train-iter', type=int, default=100000)
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
    if args.risk_extrapolation:
         args.exp_path = args.exp_path + f"risk_extrapolation_beta_{args.beta}/"
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    G1,A1 = load_graphs(args.g_path1)
    if args.g_path2 is not None:
        G2,A2 = load_graphs(args.g_path2)
    else:
        G2,A2 = G1,A1
    if args.g_path3 is not None:
        G3,A3 = load_graphs(args.g_path3)
    else:
        G3,A3 = G1,A1
    gt_adjacency = get_intersection_edges(A1,A2)
    gt_adjacency = get_intersection_edges(gt_adjacency,A3)
    dataset_train = SyntheticDataset(args.n_points, args.g_path1, args.g_path2, args.g_path3,args.sem_type, args.sigma_min,
                                args.sigma_max, args.load_data, args.data_path,device=args.device,mode='train')
    dataset_test = SyntheticDataset(args.n_points, args.g_path1, args.g_path2,args.g_path3, args.sem_type, args.sigma_min,
                                args.sigma_max, args.load_data, args.data_path,device=args.device,mode='test')
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,generator=torch.Generator(device='cuda'))
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,generator=torch.Generator(device='cuda'))

    # creating the model
    model = LearnableModel_NonLinGaussANM(args.num_vars, args.num_layers, args.hid_dim, args.nonlin,args.norm_prod,args.square_prod)
    model.to(args.device)
    # applying preliminary neighbor selection (PNS)
    if args.num_neighbors is None:
            num_neighbors = args.num_vars
    else:
        num_neighbors = args.num_neighbors
    print("Making pns folder")
    pns(model, train_loader, test_loader, num_neighbors, args.pns_thresh, args.exp_path, A1,A2,A3)


    # learning loop
    if args.train:
        train(model,gt_adjacency,A1,A2,A3,train_loader,test_loader,args)
    # making it a DAG
    if args.to_dag:
         to_dag(model,train_loader,A1,A2,A3,args)
    

    

if __name__ == '__main__':
    main()