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


def perform_knockoff_filtering(X, Y, groups, m=5, q=0.1, own_imp=False):
    """
    Perform knockoff filtering using modelX_gaussian_group_knockoffs from knockoffspy.
    
    Steps:
    1. Define groups for features.
    2. Estimate mu and Sigma from X.
    3. Generate model-X Gaussian group knockoffs for X.
    4. Fit a Lasso model on [X, X_knock1, ..., X_knockm].
    5. Compute W statistics: |beta_j| - median(|beta_knock1_j|, ..., |beta_knockm_j|).
    6. Determine κ: which among original and knockoffs has the highest |beta|.
    7. Apply mk_threshold to find the cutoff T.
    8. Select variables where W_j >= T.
    
    Returns:
    - selected: array of selected variable indices
    """
    
    if not own_imp:
        n, p = X.shape
        num_groups = len(np.unique(groups))
        
        # Estimate mu and Sigma from X
        mu = X.mean(axis=0)
        Sigma = np.cov(X, rowvar=False)
        
        # Generate group knockoffs
        solver = "maxent"  # Choices: "maxent", "mvr", "sdp", "equi"
        try:
            result = ko.modelX_gaussian_group_knockoffs(X, solver, groups, mu, Sigma, m=m, verbose=False)
        except Exception as e:
            print(f"Error generating knockoffs: {e}")
            return np.array([])
        
        X_knock = result.Xko  # Shape: (n, p * m)
        
        # Fit Lasso on [X, X_knock]
        XX = np.hstack([X, X_knock])
        lasso = Lasso(alpha=0.01, fit_intercept=True)
        lasso.fit(XX, Y)
        beta_hat = lasso.coef_  # Shape: (p + p*m,)
        
        # Compute W statistics and κ
        tau = np.zeros(p)
        kappa = np.zeros(p, dtype=int)
        
        for j in range(p):
            # Original feature coefficient
            beta_original = np.abs(beta_hat[j])
            
            # Knockoff coefficients for feature j
            knockoff_start = j + 1 + j * m
            knockoff_end = knockoff_start + m
            beta_knockoffs = np.abs(beta_hat[knockoff_start:knockoff_end])
            
            # Compute tau[j] as original - median of knockoffs
            tau[j] = beta_original - np.median(beta_knockoffs)
            
            # Determine which has the maximum coefficient
            all_betas = np.concatenate(([beta_original], beta_knockoffs))
            max_idx = np.argmax(all_betas)
            kappa[j] = max_idx  # 0 for original, 1 to m for knockoffs
        
        # Apply knockoff thresholding
        try:
            # mk_threshold expects lists
            T = ko.mk_threshold(tau.tolist(), kappa.tolist(), m, q)
        except Exception as e:
            print(f"Error computing threshold: {e}")
            return np.array([])
        
        # Select variables
        selected = np.where(tau >= T)[0]
        
        return selected


    if own_imp:
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
