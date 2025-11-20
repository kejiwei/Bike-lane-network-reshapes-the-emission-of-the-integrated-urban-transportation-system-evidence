import numpy as np
from linearmodels.iv import IVGMM

# ---- utilities ----
def utilities_from_beta(
    beta,            # [β0D, βtD, βXD..., β0C, βtC, βrhoC, βXC..., β0O, βtO, βXO...]
    XD, XC, XO,      
    tD, tC, tO, rho  # driving time, cycling time, outside time, bike-lane coverage
):
    XD, XC, XO = map(np.asarray, (XD, XC, XO))
    tD, tC, tO, rho = map(lambda v: np.asarray(v).reshape(-1), (tD, tC, tO, rho))

    kD, kC, kO = XD.shape[1],  XC.shape[1],  XO.shape[1]
    off = 0
    β0D, βtD = beta[off], beta[off+1]; off += 2
    βXD = beta[off:off+kD]; off += kD

    β0C, βtC, βrhoC = beta[off], beta[off+1], beta[off+2]; off += 3
    βXC = beta[off:off+kC]; off += kC

    β0O, βtO = beta[off], beta[off+1]; off += 2
    βXO = beta[off:off+kO]; off += kO
    if off != beta.size:
        raise ValueError("beta length mismatch.")

    # Utilities RELATIVE TO N (u_N = 0)
    uD = β0D + βtD*tD + XD @ βXD
    print(f'uD has shape {uD.shape}')
    uC = β0C + βtC*tC + βrhoC*rho + XC @ βXC
    print(f'uC has shape {uC.shape}')
    uO = β0O + βtO*tO + XO @ βXO
    print(f'uO has shape {uO.shape}')

    return uD, uC, uO


def safe_log_ratio(a, b, eps=1e-8):
    """Compute log(a/b) with small epsilon to avoid log(0)."""
    a_ = np.clip(np.asarray(a), eps, None)
    b_ = np.clip(np.asarray(b), eps, None)
    return np.log(a_) - np.log(b_)


def logsumexp3(a, b, c):
    m = np.maximum.reduce([a, b, c])
    return m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m))


def alpha_of_beta(
    beta: np.ndarray,
    d_obs: np.ndarray,          # (W,) observed total demand
    Z_dem: np.ndarray,          # (W, kZ) exogenous controls in demand eq.  (can be empty: (W, 0))
    Q_dem: np.ndarray,          # (W, Lq) instruments for IV(β)
    XD: np.ndarray, XC: np.ndarray, XO: np.ndarray,  # utility features (no time/ρ)
    tD: np.ndarray, tC: np.ndarray, tO: np.ndarray,  # times
    rho: np.ndarray,            # bike-lane coverage (for cycling utility)
    clusters: np.ndarray = None,# (W, ) cluster ids (optional)
    weight_type: str = "robust" # IVGMM weight
):
    """
    Given β, compute α(β) from the linear IV-GMM demand:
        d = α0 + α1 * IV(β) + Z_dem α2 + ε,
    where IV(β) = log(exp(uD) + exp(uC) + exp(uO)).

    Returns
    -------
    alpha : np.ndarray              # [α0, α1, αZ...]
    res   : linearmodels result     # IVGMM fit result (for SEs, diagnostics)
    IV    : np.ndarray              # (W, ) inclusive value used
    d_hat : np.ndarray              # (W, ) fitted demand
    resid : np.ndarray              # (W, ) residuals
    """
    # 1) deterministic utilities and inclusive value IV(β)
    uD, uC, uO = utilities_from_beta(beta, XD, XC, XO, tD, tC, tO, rho)
    IV = logsumexp3(uD, uC, uO)

    # 2) build regressors (handle empty Z_dem)
    d = np.asarray(d_obs).reshape(-1)
    W = d.shape[0]
    has_Z = (Z_dem is not None) and (np.asarray(Z_dem).size > 0)

    if has_Z:
        Z_dem = np.asarray(Z_dem).reshape(W, -1)
        exog = np.column_stack([np.ones(W), Z_dem])   # [const, Z_dem]
        kZ = Z_dem.shape[1]
    else:
        exog = np.ones((W, 1))                        # [const] only
        kZ = 0

    endog = IV.reshape(W, 1)                          # single endogenous regressor
    instr = np.asarray(Q_dem).reshape(W, -1)          # excluded instruments for IV(β)

    # 3) IV-GMM fit
    mod = IVGMM(dependent=d, exog=exog, endog=endog, instruments=instr)
    # fit_kw = {"weight_type": weight_type, "debiased": True}
    if clusters is not None:
        res = mod.fit(cov_type="clustered", clusters=clusters)
    else:
        res = mod.fit(cov_type="robust")
    print(f'given beta, we estimated alpha with IVGMM; summary:\n{res.summary}')

    # 4) extract parameters by POSITION (exog first, then endog)
    # exog has 1 + kZ columns; endog has 1 column
    params = res.params.values
    k_exog = exog.shape[1]        # = 1 (+ kZ)
    # positions: [α0, (αZ...)] then α1 (for IV)
    alpha0 = params[0]
    alphaZ = params[1:1 + kZ] if kZ > 0 else np.array([])
    alpha1 = params[k_exog]       # first (and only) endog coefficient

    alpha = np.concatenate([[alpha0, alpha1], alphaZ])

    # 5) fitted values & residuals
    if kZ > 0:
        d_hat = alpha0 + alpha1 * IV + Z_dem @ alphaZ
    else:
        d_hat = alpha0 + alpha1 * IV
    resid = d - d_hat

    return alpha, res, IV, d_hat, resid


def build_all_moments(
    beta,
    *,
    sD, sC, sO,                    
    XD, XC, XO,                  
    tD, tC, tO,                   
    rho,                           
    utilities_from_beta,           
    Z_D, Z_C,                      
    d_obs,                          
    Z_dem,                          
    Q_dem,                         
    alpha=None,                     # np.r_[α0, α1, αZ...]
    alpha_solver=None,              

    # ----------- Output control -----------
    return_per_obs=False
):
    """

    Share moments :
        E[ Z_D * { log(sD/sO) - (uD(β) - uO(β)) } ] = 0
        E[ Z_C * { log(sC/sO) - (uC(β) - uO(β)) } ] = 0

    Demand moments:
        E[ Q_dem * { d - (α0 + α1 * IV(β) + Z_dem' α2) } ] = 0,
    where IV(β) = log( exp(uD) + exp(uC) + exp(uO) ).
    """
    utils = utilities_from_beta(beta, XD, XC, XO, tD, tC, tO, rho)
    if isinstance(utils, (tuple, list)) and len(utils) == 4:
        uD, uC, uO, _ = utils
    else:
        uD, uC, uO = utils

    def safe_log_ratio(a, b, eps=1e-8):
        a = np.clip(np.asarray(a).reshape(-1), eps, None)
        b = np.clip(np.asarray(b).reshape(-1), eps, None)
        return np.log(a) - np.log(b)

    rD = safe_log_ratio(sD, sO) - (uD - uO)
    rC = safe_log_ratio(sC, sO) - (uC - uO)

    def logsumexp3(a, b, c):
        m = np.maximum.reduce([a, b, c])
        return m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m))

    IV = logsumexp3(uD, uC, uO)  # log-sum over {D,C,O}

    if alpha is None:
        if alpha_solver is None:
            raise ValueError("Either provide `alpha` or pass an `alpha_solver` callable.")
        solved = alpha_solver(beta, d_obs, Z_dem, Q_dem, XD, XC, XO, tD, tC, tO, rho)
        alpha = solved if isinstance(solved, np.ndarray) else solved[0]
    alpha = np.asarray(alpha).reshape(-1)
    α0, α1 = alpha[0], alpha[1]
    αZ = alpha[2:]  # may be empty if there are no demand-side controls

    has_Z = not (Z_dem is None or np.asarray(Z_dem).size == 0)
    if has_Z:
        r_dem = np.asarray(d_obs).reshape(-1) - (α0 + α1 * IV + Z_dem @ αZ)
    else:
        r_dem = np.asarray(d_obs).reshape(-1) - (α0 + α1 * IV)

    # --- 4) Stack moments ---
    if return_per_obs:
        gD  = (Z_D * rD[:, None]).reshape(-1)       # (W*L_D,)
        gC  = (Z_C * rC[:, None]).reshape(-1)       # (W*L_C,)
        gDM = (Q_dem * r_dem[:, None]).reshape(-1)  # (W*L_q,)
        g = np.concatenate([gD, gC, gDM], axis=0)
    else:
        gD  = (Z_D * rD[:, None]).mean(axis=0)      # (L_D,)
        gC  = (Z_C * rC[:, None]).mean(axis=0)      # (L_C,)
        gDM = (Q_dem * r_dem[:, None]).mean(axis=0) # (L_q,)
        g = np.concatenate([gD, gC, gDM], axis=0)

    return g
