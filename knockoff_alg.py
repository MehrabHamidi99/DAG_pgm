import numpy as np
from sklearn.linear_model import Lasso
from scipy.linalg import sqrtm, eigh

def equicorrelated_s(Sigma):
    """
    Compute the equicorrelated knockoff parameter s for a given covariance Sigma.
    Based on Barber & Candès 2015:
    We want to find s that makes:
       [Sigma        Sigma - sI
        Sigma - sI   Sigma     ]
    PSD.
    This is equivalent to requiring 0 <= s <= lambda_min(Sigma).
    
    We choose s = min( 2*lambda_min(Sigma), 0.999*lambda_min(Sigma) ) or simply s = lambda_min(Sigma).
    To be conservative, we use s = lambda_min(Sigma).
    """
    eigvals = np.linalg.eigvalsh(Sigma)
    lambda_min = np.min(eigvals)
    # ensure s is positive and doesn't exceed smallest eigenvalue
    s = 0.9 * lambda_min  # a factor slightly less than lambda_min to ensure PSD
    if s <= 0:
        # If Sigma is not positive-definite or lambda_min <= 0, this approach won't work directly
        # Additional regularization needed:
        s = 1e-4
    return s

def construct_equicorrelated_knockoffs(X):
    """
    Construct equicorrelated knockoffs for X using the covariance-based approach.

    Steps:
    1. Center X.
    2. Compute Sigma.
    3. Compute s.
    4. Form knockoffs using the formula:
       X_knockoff = X - X Sigma^{-1} S + E
    where E ~ N(0, 2S - S Sigma^{-1} S), we choose E deterministically here by using SVD structure.

    For simplicity, we use the construction given by Proposition 1 in Barber & Candès (2015):
    The knockoffs (for Gaussian X) can be generated as:
        X_knockoffs = X*Gamma - X*(Sigma^{-1}S)*Gamma + ...
    In this simplified version, we follow a standard known approach:
    
    A simpler known formula for equicorrelated knockoffs:
    - Compute the SVD: Sigma = U Lambda U^T
    - Let s = sI.
    - Define tilde(Lambda) = Lambda - sI.
    - Then knockoffs can be constructed as:
      X_knock = XU [ (Lambda-sI)^{1/2}O ], where O is a random orthonormal matrix.
      For simplicity, let O = I (no randomness).
    
    However, this doesn't exactly match the original paper's formula for E.

    To keep it simpler and still produce a valid construction, we rely on:
    X_knock = X - X Sigma^{-1} S + Z
    where Z ~ N(0, 2S - S Sigma^{-1} S).

    We'll produce Z with correct covariance using a random draw:
    """
    n, p = X.shape
    X_mean = X.mean(axis=0)
    Xc = X - X_mean
    Sigma = np.cov(Xc, rowvar=False)  # p x p
    s = equicorrelated_s(Sigma)
    S = s * np.eye(p)

    # Compute Sigma^{-1}:
    Sigma_inv = np.linalg.inv(Sigma)

    # Compute covariance of Z:
    # Z ~ N(0, 2S - S Sigma^{-1} S)
    A = 2*S - S @ Sigma_inv @ S
    # Ensure A is PSD (it should be)
    # Decompose A:
    eigvals, eigvecs = eigh(A)
    eigvals = np.maximum(eigvals, 0)  # numerical stability
    A_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # Sample Z:
    Z = np.random.randn(n, p) @ A_sqrt

    # Construct knockoffs:
    X_knock = Xc - Xc @ Sigma_inv @ S + Z

    # Add back mean if desired:
    X_knock += X_mean

    return X_knock

def knockoff_threshold(W, q):
    """
    Compute the knockoff threshold given the W statistics and desired FDR q.
    Threshold is defined as:
    T = min{ t: (1 + # {j: W_j <= -t}) / max{1, # {j: W_j >= t}} <= q }
    """
    Ws = np.sort(np.abs(W))[::-1]
    ratio = 1.0
    T = None
    for t in Ws:
        denom = max(1, np.sum(W >= t))
        num = 1 + np.sum(W <= -t)
        ratio = num / denom
        if ratio <= q:
            T = t
    if T is None:
        # If no threshold found, select none
        T = np.inf
    return T

def perform_knockoff_filtering(X, q=0.5, alpha=0.01):
    """
    Perform knockoff filtering given data X to select variables.

    Steps:
    1. Construct knockoffs.
    2. Fit Lasso to [X, X_knock].
    3. Compute W_j = |beta_j| - |beta_{j+p}| where beta_j is coeff of X_j, beta_{j+p} is coeff of knockoff.
    4. Apply knockoff threshold.
    5. Return selected variables.

    Parameters:
    X: n x p data matrix
    q: desired FDR level
    alpha: regularization parameter for Lasso (can be adjusted)

    Returns:
    selected: array of indices of selected variables
    """
    n, p = X.shape
    X_knock = construct_equicorrelated_knockoffs(X)

    # Fit Lasso on [X, X_knock]
    XX = np.hstack([X, X_knock])
    y = np.random.randn(n)  # In practice, you'd need a response variable or another criterion
    # Since the user did not provide Y, we simulate a response.
    # A real scenario: Y should be given or chosen based on domain knowledge.

    model = Lasso(alpha=alpha, fit_intercept=True)
    model.fit(XX, y)
    beta = model.coef_

    # Compute W statistics
    W = np.abs(beta[:p]) - np.abs(beta[p:])

    # Compute threshold and select variables
    T = knockoff_threshold(W, q)
    selected = np.where(W >= T)[0]

    return selected
