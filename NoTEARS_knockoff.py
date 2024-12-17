import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import knockpy
from knockpy import KnockoffFilter
from sklearn.preprocessing import StandardScaler

def notears_linear_knockoff(X, lambda1, loss_type, fdr=0.1, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 + knockoff_penalty(W) s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        fdr (float): target false discovery rate for knockoff-based regularizer
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
            G_loss = -1.0 / X.shape[0] * X.T @ R
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
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _knockoff_penalty(W):
        """Compute knockoff-based penalty for the adjacency matrix W."""
        d = W.shape[0]
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Generate knockoff features for the dataset
        Sigma = knockpy.dgp.AR1(p=d, rho=0.5)
        kfilter = KnockoffFilter(ksampler='gaussian', fstat='lasso')
        knockoffs = kfilter.sample_knockoffs(X_std, Sigma=Sigma)

        # Calculate knockoff statistics
        W_flat = W.flatten()
        W_knockoff = knockoffs @ W_flat.reshape(d, d)
        knockoff_stats = np.abs(W_flat) - np.abs(W_knockoff.flatten())

        # Penalty for weights below the threshold
        threshold = np.percentile(knockoff_stats, (1 - fdr) * 100)
        penalty = np.sum(np.maximum(0, threshold - knockoff_stats))

        return penalty

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        knockoff_penalty = _knockoff_penalty(W)

        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum() + knockoff_penalty
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
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
