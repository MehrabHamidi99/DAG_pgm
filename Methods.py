import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import knockpy
from knockpy import KnockoffFilter
from sklearn.preprocessing import StandardScaler

def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est



def notears_knockoff_regularizer(X, lambda1, loss_type, fdr=0.1, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    
    def _knockoff_penalty(W):
        """Compute knockoff-based penalty for the adjacency matrix W."""
        d = W.shape[0]
        scaler = StandardScaler()
        # X_std = scaler.fit_transform(X)

        # Generate knockoff features for the dataset
        Sigma = knockpy.dgp.AR1(p=d, rho=0.5)
        kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
        # Run knockoff filter to select significant features
        # rejections = kfilter.forward(
        #     X=X,
        #     y=np.zeros(X.shape[0]),  # Random target for unsupervised setting
        #     Sigma=Sigma,
        #     fdr=0.1
        # )

        # Preliminaries - infer covariance matrix for MX
        if Sigma is None and kfilter._mx:
            Sigma, _ = utilities.estimate_covariance(X, 1e-2, shrinkage)
            # Possible factor model approximation
            if num_factors is not None and Sigma is not None:
                kfilter.D, kfilter.U = utilities.estimate_factor(
                    Sigma, num_factors=num_factors
                )
                Sigma = np.diag(kfilter.D) + np.dot(kfilter.U, kfilter.U.T)
                kfilter.knockoff_kwargs['how_approx'] = 'factor'
                kfilter.knockoff_kwargs['D'] = kfilter.D
                kfilter.knockoff_kwargs['U'] = kfilter.U
            else:
                kfilter.D = None
                kfilter.U = None
        if not kfilter._mx:
            Sigma = None


        # Save objects
        kfilter.X = X
        kfilter.Xk = None
        # Center if we are going to fit FX knockoffs
        if kfilter.Xk is None and not kfilter._mx:
            kfilter.X = kfilter.X - kfilter.X.mean(axis=0)
        kfilter.y = None
        kfilter.mu = None
        kfilter.Sigma = Sigma
        kfilter.groups = None
        # for key in fstat_kwargs:
        #     kfilter.fstat_kwargs[key] = fstat_kwargs[key]
        # for key in knockoff_kwargs:
        #     kfilter.knockoff_kwargs[key] = knockoff_kwargs[key]

        # Save n, p, groups
        n = X.shape[0]
        p = X.shape[1]
        if kfilter.groups is None:
            kfilter.groups = np.arange(1, p + 1, 1)

        recycle_up_to = None

        # Parse recycle_up_to
        if recycle_up_to is None:
            pass
        elif recycle_up_to < 1:
            recycle_up_to = int(recycle_up_to * n)
        else:
            recycle_up_to = int(recycle_up_to)
        kfilter.recycle_up_to = recycle_up_to

        knockoffs = kfilter.sample_knockoffs()
        S = kfilter.ksampler.fetch_S()

        # Calculate knockoff statistics
        W_flat = W.flatten()
        W_knockoff = knockoffs @ W_flat.reshape(d, d)
        knockoff_stats = np.abs(W_flat) - np.abs(S.flatten())

        penalty = np.linalg.norm(knockoff_stats)

        # # Penalty for weights below the threshold
        # threshold = np.percentile(knockoff_stats, (1 - fdr) * 100)
        # penalty = np.sum(np.maximum(0, threshold - knockoff_stats))

        return penalty

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)

        knockoff_penalty = _knockoff_penalty(W)

        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum() + 0.075 * knockoff_penalty
        # print(obj)
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        # print(_)
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# # coding=utf-8
# """
# GraN-DAG

# Copyright © 2019 Authors of Gradient-Based Neural DAG Learning

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# """
# import os
# import pickle

# import torch
# import torch.nn as nn
# import torch.nn.functional as F



# def compute_A_phi(model, norm="none", square=False): # Method to compute the weighted adjacency matrix from the output of the d models
#     weights = model.get_parameters(mode='w')[0]
#     prod = torch.eye(model.num_vars)
#     if norm != "none":
#         prod_norm = torch.eye(model.num_vars)
#     for i, w in enumerate(weights):
#         if square:
#             w = w ** 2
#         else:
#             w = torch.abs(w)
#         if i == 0:
#             prod = torch.einsum("tij,ljt,jk->tik", w, model.adjacency.unsqueeze(0), prod)
#             if norm != "none":
#                 tmp = 1. - torch.eye(model.num_vars).unsqueeze(0)
#                 prod_norm = torch.einsum("tij,ljt,jk->tik", torch.ones_like(w).detach(), tmp, prod_norm)
#         else:
#             prod = torch.einsum("tij,tjk->tik", w, prod)
#             if norm != "none":
#                 prod_norm = torch.einsum("tij,tjk->tik", torch.ones_like(w).detach(), prod_norm)

#     # sum over density parameter axis
#     prod = torch.sum(prod, 1)
#     if norm == "paths":
#         prod_norm = torch.sum(prod_norm, 1)
#         denominator = prod_norm + torch.eye(model.num_vars)  # avoid / 0 on diagonal
#         return (prod / denominator).t()
#     elif norm == "none":
#         return prod.t()
#     else:
#         raise NotImplementedError


# class BaseModel(nn.Module):
#     def __init__(self, num_vars, num_layers, hid_dim, num_params, nonlin="leaky-relu", norm_prod='path',
#                  square_prod=False):
#         """

#         :param num_vars: number of variables in the system
#         :param num_layers: number of hidden layers
#         :param hid_dim: number of hidden units per layer
#         :param num_params: number of parameters per conditional *outputted by MLP*
#         :param nonlin: which nonlinearity
#         """
#         super(BaseModel, self).__init__()
#         self.num_vars = num_vars
#         self.num_layers = num_layers
#         self.hid_dim = hid_dim
#         self.num_params = num_params
#         self.nonlin = nonlin
#         self.norm_prod = norm_prod
#         self.square_prod = square_prod

#         self.weights = nn.ParameterList()
#         self.biases = nn.ParameterList()
#         self.extra_params = []  # Those parameter might be learnable, but they do not depend on parents.

#         # initialize current adjacency matrix
#         self.adjacency = torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars)

#         self.zero_weights_ratio = 0.
#         self.numel_weights = 0

#         # Instantiate the parameters of each layer in the model of each variable
#         for i in range(self.num_layers + 1):
#             in_dim = self.hid_dim
#             out_dim = self.hid_dim
#             if i == 0:
#                 in_dim = self.num_vars
#             if i == self.num_layers:
#                 out_dim = self.num_params
#             self.weights.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, in_dim)))
#             self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim)))
#             self.numel_weights += self.num_vars * out_dim * in_dim

#     def forward_given_params(self, x, weights, biases):
#         """

#         :param x: batch_size x num_vars
#         :param weights: list of lists. ith list contains weights for ith MLP
#         :param biases: list of lists. ith list contains biases for ith MLP
#         :return: batch_size x num_vars * num_params, the parameters of each variable conditional
#         """
#         bs = x.size(0)
#         num_zero_weights = 0
#         for k in range(self.num_layers + 1):
#             # apply affine operator
#             if k == 0:
#                 adj = self.adjacency.unsqueeze(0)
#                 x = torch.einsum("tij,ljt,bj->bti", weights[k], adj, x) + biases[k]
#             else:
#                 x = torch.einsum("tij,btj->bti", weights[k], x) + biases[k]

#             # count num of zeros
#             num_zero_weights += weights[k].numel() - weights[k].nonzero().size(0)

#             # apply non-linearity
#             if k != self.num_layers:
#                 x = F.leaky_relu(x) if self.nonlin == "leaky-relu" else torch.sigmoid(x)

#         self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)

#         return torch.unbind(x, 1)

#     def get_w_adj(self):
#         """Get weighted adjacency matrix"""
#         return compute_A_phi(self, norm=self.norm_prod, square=self.square_prod)

#     def reset_params(self):
#         with torch.no_grad():
#             for node in range(self.num_vars):
#                 for i, w in enumerate(self.weights):
#                     w = w[node]
#                     nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu'))
#                 for i, b in enumerate(self.biases):
#                     b = b[node]
#                     b.zero_()

#     def get_parameters(self, mode="wbx"):
#         """
#         Will get only parameters with requires_grad == True
#         :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
#         :return: corresponding dicts of parameters
#         """
#         params = []

#         if 'w' in mode:
#             weights = []
#             for w in self.weights:
#                 weights.append(w)
#             params.append(weights)
#         if 'b'in mode:
#             biases = []
#             for j, b in enumerate(self.biases):
#                 biases.append(b)
#             params.append(biases)

#         if 'x' in mode:
#             extra_params = []
#             for ep in self.extra_params:
#                 if ep.requires_grad:
#                     extra_params.append(ep)
#             params.append(extra_params)

#         return tuple(params)

#     def set_parameters(self, params, mode="wbx"):
#         """
#         Will set only parameters with requires_grad == True
#         :param params: tuple of parameter lists to set, the order should be coherent with `get_parameters`
#         :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
#         :return: None
#         """
#         with torch.no_grad():
#             k = 0
#             if 'w' in mode:
#                 for i, w in enumerate(self.weights):
#                     w.copy_(params[k][i])
#                 k += 1

#             if 'b' in mode:
#                 for i, b in enumerate(self.biases):
#                     b.copy_(params[k][i])
#                 k += 1

#             if 'x' in mode and len(self.extra_params) > 0:
#                 for i, ep in enumerate(self.extra_params):
#                     if ep.requires_grad:
#                         ep.copy_(params[k][i])
#                 k += 1

#     def get_grad_norm(self, mode="wbx"):
#         """
#         Will get only parameters with requires_grad == True, simply get the .grad
#         :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
#         :return: corresponding dicts of parameters
#         """
#         grad_norm = 0

#         if 'w' in mode:
#             for w in self.weights:
#                 grad_norm += torch.sum(w.grad ** 2)

#         if 'b'in mode:
#             for j, b in enumerate(self.biases):
#                 grad_norm += torch.sum(b.grad ** 2)

#         if 'x' in mode:
#             for ep in self.extra_params:
#                 if ep.requires_grad:
#                     grad_norm += torch.sum(ep.grad ** 2)

#         return torch.sqrt(grad_norm)

#     def save_parameters(self, exp_path, mode="wbx"):
#         params = self.get_parameters(mode=mode)
#         # save
#         with open(os.path.join(exp_path, "params_"+mode), 'wb') as f:
#             pickle.dump(params, f)

#     def load_parameters(self, exp_path, mode="wbx"):
#         with open(os.path.join(exp_path, "params_"+mode), 'rb') as f:
#             params = pickle.load(f)
#         self.set_parameters(params, mode=mode)

#     def get_distribution(self, density_params):
#         raise NotImplementedError
