import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
from dag_optim import compute_constraint,compute_jacobian_avg,is_acyclic
from plot import plot_adjacency, plot_learning_curves,plot_adjacency_intersections
from utils import dump
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

EPSILON=1E-8

def pns_(model_adj, train_loader, test_loader, num_neighbors, thresh):
    """Preliminary neighborhood selection"""
    x_train, _ = next(iter(train_loader))
    x_test, _ = next(iter(test_loader))
    x = np.concatenate([x_train.detach().cpu().numpy(), x_test.detach().cpu().numpy()], 0)

    num_samples = x.shape[0]
    num_nodes = x.shape[1]
    print("PNS: num samples = {}, num nodes = {}".format(num_samples, num_nodes))
    for node in range(num_nodes):
        print("PNS: node " + str(node))
        x_other = np.copy(x)
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(x_other, x[:, node])
        selected_reg = SelectFromModel(reg, threshold="{}*mean".format(thresh), prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False).astype(float)

        model_adj[:, node] *= mask_selected

    return model_adj


def pns(model, train_loader, test_loader, num_neighbors, thresh, exp_path, A1,A2,A3):
    # Prepare path for saving results
    save_path = os.path.join(exp_path, "pns")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_adj = model.adjacency.detach().cpu().numpy()
    time0 = time.time()
    model_adj = pns_(model_adj, train_loader, test_loader, num_neighbors, thresh)

    with torch.no_grad():
        model.adjacency.copy_(torch.Tensor(model_adj))

    timing = time.time() - time0
    print("PNS took {}s".format(timing))
    # save
    dump(model, save_path, 'model')
    dump(timing, save_path, 'timing')
    np.save(os.path.join(save_path, "DAG"), model.adjacency.detach().cpu().numpy())
    # plot
    plot_adjacency_intersections(model.adjacency.detach().cpu().numpy(), A1, A2, save_path, name='_pns_intersections',gt_3=A3)
    return model






def train(model,gt_adjacency,adjacency1,adjacency2,adjacency3,train_loader,test_loader,args):
    save_path = os.path.join(args.exp_path, 'train')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    time0 = time.time()
    # initialize stuff for learning loop
    aug_lagrangians = []
    aug_lagrangian_ma = [0.0] * (args.num_train_iter + 1)
    aug_lagrangians_val = []
    grad_norms = []
    grad_norm_ma = [0.0] * (args.num_train_iter + 1)
    variances = []
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
        log_like = model.compute_log_likelihood(x, weights, biases, extra_params)
        loss = - torch.mean(log_like)
        if args.risk_extrapolation:
            mask_0 = (label == 0)
            mask_1 = (label == 1)
            per_sample_loss = -log_like
            mean_loss_0 = torch.mean(per_sample_loss[mask_0]) if mask_0.any() else torch.tensor(0.0, device=per_sample_loss.device)
            mean_loss_1 = torch.mean(per_sample_loss[mask_1]) if mask_1.any() else torch.tensor(0.0, device=per_sample_loss.device)
            if args.g_path3 is not None:
                mask_2 = (label == 2)
                mean_loss_2 = torch.mean(per_sample_loss[mask_2]) if mask_2.any() else torch.tensor(0.0, device=per_sample_loss.device)
                group_means = torch.stack([mean_loss_0, mean_loss_1,mean_loss_2])
            else:
                group_means = torch.stack([mean_loss_0, mean_loss_1])
            variance_of_means = torch.var(group_means, unbiased=True)
            variances.append(variance_of_means.item())
            loss += args.beta * variance_of_means
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
            plot_adjacency_intersections(model.adjacency.detach().cpu().numpy(), adjacency1, adjacency2, args.exp_path,name='_intersections',gt_3=adjacency3)
            if variances:
                if variances == []:
                    variances = None
            plot_learning_curves(not_nlls, aug_lagrangians, aug_lagrangian_ma[:iteration], aug_lagrangians_val, nlls,
                                 nlls_val, args.exp_path,variances)
        

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
            plot_adjacency_intersections(model.adjacency.detach().cpu().numpy(), adjacency1, adjacency2, save_path,name='_intersections',gt_3=adjacency3)
            if variances:
                if len(variances)==0:
                    variances = None
            plot_learning_curves(not_nlls, aug_lagrangians, aug_lagrangian_ma[:iteration], aug_lagrangians_val, nlls,
                                 nlls_val, save_path,variances)

            return model
        


def to_dag(model, train_loader, A1,A2,A3, args, stage_name="to-dag"):
    """
    1- If some entries of A_\phi == 0, also mask them (This can happen with stochastic proximal gradient descent)
    2- Remove edges (from weaker to stronger) until a DAG is obtained.

    """
    # Prepare path for saving results
    save_path = os.path.join(args.exp_path, stage_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.eval()
    time0 = time.time()

    if args.jac_thresh:
        A = compute_jacobian_avg(model, train_loader, args.batch_size).t()
    else:
        A = model.get_w_adj()
    A = A.detach().cpu().numpy()

    with torch.no_grad():
        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(A)
        for step, t in enumerate(thresholds):
            print("Edges/thresh", model.adjacency.sum(), t)
            to_keep = torch.Tensor(A > t + EPSILON)
            new_adj = model.adjacency * to_keep

            if is_acyclic(new_adj.cpu()):
                model.adjacency.copy_(new_adj)
                break


    # Save
    dump(model, save_path, 'model')
    dump(args, save_path, 'opt')
    np.save(os.path.join(save_path, "DAG"), model.adjacency.detach().cpu().numpy())
    # plot adjacency
    plot_adjacency_intersections(model.adjacency.detach().cpu().numpy(), A1, A2, save_path, name='_intersections',gt_3=A3)
    return model


