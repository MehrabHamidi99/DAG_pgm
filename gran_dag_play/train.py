import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
from dag_optim import compute_constraint
from plot import plot_adjacency, plot_learning_curves
from utils import dump





def train(model,gt_adjacency,train_loader,test_loader,args):
    save_path = os.path.join(args.output_ckpt, 'train')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    time0 = time.time()
    # initialize stuff for learning loop
    aug_lagrangians = []
    aug_lagrangian_ma = [0.0] * (args.num_train_iter + 1)
    aug_lagrangians_val = []
    grad_norms = []
    grad_norm_ma = [0.0] * (args.num_train_iter + 1)

    hs = []
    not_nlls = []  # Augmented Lagrangrian minus (pseudo) NLL
    nlls = []  # NLL on train
    nlls_val = []  # NLL on validation
    delta_mu = np.inf

    # Augmented Lagrangian stuff
    mu = args.mu_init
    lamb = args.lambda_init
    mus = []
    lambdas = []

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # learning loop
    for iteration in range(args.num_train_iter):
        model.train()
        x,label = next(iter(train_loader))
        weights,biases,extra_params = model.get_parameters(mode="wbx")
        loss = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params))
        nlls.append(loss.item())
        model.eval()
        w_adj = model.get_w_adj()
        h = compute_constraint(model, w_adj)
        aug_lagrangian = loss + 0.5*mu*h**2 + lamb*h

        optimizer.zero_grad()
        aug_lagrangian.backward()
        optimizer.step()

         # clamp edges
        if args.edge_clamp_range != 0:
            with torch.no_grad():
                to_keep = (w_adj > args.edge_clamp_range).type(torch.Tensor)
                model.adjacency *= to_keep

        mus.append(mu)
        lambdas.append(lamb)
        not_nlls.append(0.5 * mu * h.item() ** 2 + lamb*h.item())

        # compute augmented lagrangian moving average
        aug_lagrangians.append(aug_lagrangian.item())
        aug_lagrangian_ma[iteration+ 1] = aug_lagrangian_ma[iteration] + 0.01 * (aug_lagrangian.item() - aug_lagrangian_ma[iteration])
        grad_norms.append(model.get_grad_norm("wbx").item())
        grad_norm_ma[iteration+ 1] = grad_norm_ma[iteration] + 0.01 * (grad_norms[-1] - grad_norm_ma[iteration])

        # compute loss on whole validation set
        if iteration% args.stop_crit_win == 0:
            with torch.no_grad():
                x, _ = next(iter(test_loader))
                loss_val = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params)).item()
                nlls_val.append(loss_val)
                aug_lagrangians_val.append([iteration, loss_val + not_nlls[-1]])
        
        # compute delta for lambda
        if iteration>= 2 * args.stop_crit_win and iteration% (2 * args.stop_crit_win) == 0:
            t0, t_half, t1 = aug_lagrangians_val[-3][1], aug_lagrangians_val[-2][1], aug_lagrangians_val[-1][1]

            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if not (min(t0, t1) < t_half < max(t0, t1)):
                delta_lambda = -np.inf
            else:
                delta_lambda = (t1 - t0) / args.stop_crit_win
        else:
            delta_lambda = -np.inf  # do not update lambda nor mu
        
        if iteration% args.plot_freq == 0:
            plot_adjacency(model.adjacency.detach().cpu().numpy(), gt_adjacency, args.exp_path)
            plot_learning_curves(not_nlls, aug_lagrangians, aug_lagrangian_ma[:iteration], aug_lagrangians_val, nlls,
                                 nlls_val, args.exp_path)
        

        if h > args.h_threshold:
            # if we have found a stationary point of the augmented loss
            if abs(delta_lambda) < args.omega_lambda or delta_lambda > 0:
                lamb += mu * h.item()
                print("Updated lambda to {}".format(lamb))

                # Did the constraint improve sufficiently?
                hs.append(h.item())
                if len(hs) >= 2:
                    if hs[-1] > hs[-2] * args.omega_mu:
                        mu *= 10
                        print("Updated mu to {}".format(mu))

                # little hack to make sure the moving average is going down.
                with torch.no_grad():
                    gap_in_not_nll = 0.5 * mu * h.item() ** 2 + lamb * h.item() - not_nlls[-1]
                    aug_lagrangian_ma[iteration+ 1] += gap_in_not_nll
                    aug_lagrangians_val[-1][1] += gap_in_not_nll

                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_reinit)

        else:
            timing = time.time() - time0

            # Final clamping of all edges == 0
            with torch.no_grad():
                to_keep = (w_adj > 0).type(torch.Tensor)
                model.adjacency *= to_keep

            # compute nll on train and validation set
            weights, biases, extra_params = model.get_parameters(mode="wbx")
            x, _ = next(iter(train_loader))
            # Since we do not have a DAG yet, this is not really a negative log likelihood.
            nll_train = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params))

            x, _ = next(iter(test_loader))
            nll_val = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params))


            # Save everything
            dump(model, save_path, 'model')
            dump(args, save_path, 'opt')
            dump(nll_train, save_path, 'pseudo-nll-train')
            dump(nll_val, save_path, 'pseudo-nll-val')
            dump(not_nlls, save_path, 'not-nlls')
            dump(aug_lagrangians, save_path, 'aug-lagrangians')
            dump(aug_lagrangian_ma[:iteration], save_path, 'aug-lagrangian-ma')
            dump(aug_lagrangians_val, save_path, 'aug-lagrangians-val')
            dump(grad_norms, save_path, 'grad-norms')
            dump(grad_norm_ma[:iteration], save_path, 'grad-norm-ma')
            dump(timing, save_path, 'timing')
            np.save(os.path.join(save_path, "DAG"), model.adjacency.detach().cpu().numpy())

            # plot
            plot_adjacency(model.adjacency.detach().cpu().numpy(), gt_adjacency, save_path)
            plot_learning_curves(not_nlls, aug_lagrangians, aug_lagrangian_ma[:iteration], aug_lagrangians_val, nlls,
                                 nlls_val, save_path)

            return model