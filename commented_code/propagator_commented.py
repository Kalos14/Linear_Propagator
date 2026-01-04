"""Core propagator-model utilities (Patzelt/Bouchaud code).

This module contains:
- empirical response functions (xcorr-based)
- calibration routines for TIM1 / TIM2 / HDIM2 kernels
- simulators for those models via fast FFT convolution

Dependencies:
- scorr: FFT-based (cross-)correlation utilities used throughout the original repo.
"""

import numpy as np
from scipy.linalg import solve_toeplitz, solve
from scipy.signal import fftconvolve
from scipy.interpolate import Rbf
from scorr import xcorr, xcorr_grouped_df, xcorrshift, fftcrop, corr_mat


# Helpers
# =====================================================================

# -----------------------------------------------------------------------------
# integrate(x)
# Purpose:
#   Convert a "differential" series into its lag-1 cumulative sum.
#   Typical use cases in this repo:
#     - turning a return series into a price series (up to a constant), or
#     - turning a differential impact kernel g(ℓ) into an integrated propagator G(ℓ).
# Inputs:
#   x : 1D array-like
# Output:
#   1D numpy array with y[0]=0 and y[t]=sum_{i< t} x[i]
# Notes:
#   The convention y[0]=0 makes the transform causal and aligns with how
#   returns are defined as one-step differences in the papers.
# -----------------------------------------------------------------------------
def integrate(x):
    "Return lag 1 sum, i.e. price from return, or an integrated kernel."
    return np.concatenate([[0], np.cumsum(x[:-1])])
    
    
# -----------------------------------------------------------------------------
# smooth_tail_rbf(k, l0=3, tau=5, smooth=1, epsilon=1)
# Purpose:
#   Regularize / denoise the long-lag tail of an estimated kernel (impact or
#   response) by fitting a smooth radial-basis-function (RBF) interpolant on
#   log-lags, while keeping the short-lag part close to the raw estimate.
# Inputs:
#   k      : 1D array-like kernel to smooth
#   l0     : index from which we consider the "tail"
#   tau    : blending time-scale; larger => preserve raw kernel longer
#   smooth : RBF smoothing parameter (SciPy)
#   epsilon: RBF shape parameter
# Output:
#   knew : smoothed kernel (same shape as k)
# Notes:
#   This is a practical stabilizer. It can bias the true long-horizon decay if
#   the raw estimate is already reliable, but it often improves conditioning.
# -----------------------------------------------------------------------------
def smooth_tail_rbf(k, l0=3, tau=5, smooth=1, epsilon=1):
    """Smooth tail of array k with radial basis functions"""
    # interpolate in log-lags
    l = np.log(np.arange(l0,len(k)))
    # estimate functions
    krbf = Rbf(
        l, k[l0:], function='multiquadric', smooth=smooth, epsilon=epsilon
    )
    # weights to blend with original for short lags
    w = np.exp(-np.arange(1,len(k)-l0+1)/ float(tau))
    # interpolate
    knew     = np.empty_like(k)
    knew[:l0]  = k[:l0]
    knew[l0:] = krbf(l) * (1-w) + k[l0:] * w
    #done
    return knew
    
# -----------------------------------------------------------------------------
# propagate(s, G, sfunc=np.sign)
# Purpose:
#   Core simulator: compute the linear propagator convolution of a sign series
#   with a kernel (TIM1, and as a building block for TIM2/HDIM2).
# Model form:
#   p_t = (G * s)_t   implemented as a fast FFT-based convolution.
# Inputs:
#   s     : 1D array-like (trade signs or signed event indicators)
#   G     : 1D array-like kernel (integrated or differential, depending on usage)
#   sfunc : optional preprocessing applied to s (default: np.sign)
# Output:
#   p : 1D numpy array of length len(s)
# Notes:
#   - This uses linear convolution via fftconvolve and then truncates to the
#     original length.
#   - Causality is your responsibility: kernels should be defined with the
#     correct lag convention (typically starting at lag 0/1).
# -----------------------------------------------------------------------------
def propagate(s, G, sfunc=np.sign):
    """Simulate propagator model from signs and one kernel.
    Equivalent to tim1, one of the kernels in tim2 or hdim2.
    """
    steps = len(s)
    s  = sfunc(s[:len(s)])
    p = fftconvolve(s, G)[:steps]
    return p

# Responses
# =====================================================================

# -----------------------------------------------------------------------------
# _return_response(ret, x, maxlag)
# Purpose:
#   Internal formatting helper used by response() and response_grouped_df().
#   It reorders the cross-correlation output into a symmetric lag axis and
#   optionally returns:
#     - lags ℓ in [-maxlag, ..., +maxlag]
#     - differential response S(ℓ)
#     - cumulative/bare response R(ℓ) = sum_{i<=ℓ} S(i)
#
# Key convention:
#   scorr.xcorr returns values ordered as:
#     [0, 1, 2, ..., maxlag, -maxlag, ..., -1]
#   This helper rearranges to:
#     [-maxlag, ..., -1, 0, 1, ..., +maxlag]
# -----------------------------------------------------------------------------
def _return_response(ret, x, maxlag):
    """Helper for response and response_grouped_df."""
    # return what?
    ret = ret.lower()
    res = []
    for i in ret:
        if i   == 'l':
            # lags
            res.append(np.arange(-maxlag,maxlag+1))
        elif i == 's':
            res.append(
                # differential response
                np.concatenate([x[-maxlag:], x[:maxlag+1]])
            )
        elif i == 'r':    
            res.append(
            # bare response === cumulated differential response
                np.concatenate([
                    -np.cumsum(x[:-maxlag-1:-1])[::-1], 
                    [0], 
                    np.cumsum(x[:maxlag])
                ])
            )
    if len(res) > 1:
        return tuple(res)
    else:
        return res[0]

# -----------------------------------------------------------------------------
# response(r, s, maxlag=1e4, ret='lsr', subtract_mean=False)
# Purpose:
#   Compute the empirical price response between returns r(t) and a sign-like
#   series s(t) using a cross-covariance (xcorr).
#
# Output objects (common in the papers):
#   S(ℓ): "incremental/differential" response at lag ℓ
#   R(ℓ): "bare/cumulative" response (cumsum of S)
#
# Notes:
#   The docstring correctly warns this is NOT "linear response theory" from
#   physics; it's simply the market microstructure response definition.
# -----------------------------------------------------------------------------
def response(r, s, maxlag=10**4, ret='lsr', subtract_mean=False):
    """Return lag, differential response S, response R.
    
    Note that this commonly used price response is a simple cross correlation 
    and NOT equivalent to the linear response in systems analysis.
    
    Parameters:
    ===========
    
    r: array-like
        Returns
    s: array-like
        Order signs
    maxlag: int
        Longest lag to calculate
    ret: str
        can include 'l' to return lags, 'r' to return response, and
        's' to return differential response (in specified order).
    subtract_mean: bool
        Subtract means first. Default: False (signal means already zero)
    """
    maxlag = min(maxlag, len(r) - 2)
    s  = s[:len(r)]
    # diff. resp.
    # xcorr == S(0, 1, ..., maxlag, -maxlag, ... -1)
    x = xcorr(r, s, norm='cov', subtract_mean=subtract_mean)
    return _return_response(ret, x, maxlag)

# -----------------------------------------------------------------------------
# response_grouped_df(df, cols, nfft='pad', ...)
# Purpose:
#   Same as response(), but computed in a grouped/Welch-like manner:
#   the data are split by 'date' and correlations are averaged across groups.
#
# Why this matters:
#   - prevents artificial wrap-around across day boundaries when using FFTs
#   - improves signal-to-noise by averaging independent daily estimates
#
# Important implementation detail:
#   maxlag is inferred as len(x)/2 (Python 2 integer division). In Python 3 you
#   may want `maxlag = len(x) // 2` to keep it an int.
# -----------------------------------------------------------------------------
def response_grouped_df(
        df, cols, nfft='pad', ret='lsr', subtract_mean=False, **kwargs
    ):
    """Return lag, differential response S, response R calculated daily.
    
    Note that this commonly used price response is a simple cross correlation 
    and NOT equivalent to the linear response in systems analysis.
    
    Parameters
    ==========
    
    df: pandas.DataFrame
        Dataframe containing order signs and returns
    cols: tuple
        The columns of interest
    nfft:
        Length of the fft segments
    ret: str
        What to return ('l': lags, 'r': response, 's': incremental response).
    subtract_mean: bool
        Subtract means first. Default: False (signal means already zero)
    
    See also response, spectral.xcorr_grouped_df for more explanations
    """
    # diff. resp.
    x = xcorr_grouped_df(
        df, 
        cols,
        by            = 'date', 
        nfft          = nfft, 
        funcs         = (lambda x: x, lambda x: x), 
        subtract_mean = subtract_mean,
        norm          = 'cov',
        return_df     = False,
        **kwargs
    )[0]
    # lag 1 -> element 0, lag 0 -> element -1, ...
    #x = x['xcorr'].values[x.index.values-1]
    maxlag = len(x) / 2  # NOTE: Python 3: use `len(x) // 2` to keep this an int
    return _return_response(ret, x, maxlag)
   
# Analytical power-laws
# =====================================================================

# -----------------------------------------------------------------------------
# beta_from_gamma(gamma)
# Purpose:
#   Analytical helper: in the long-memory order-flow story, if sign
#   autocorrelation decays as C(ℓ) ~ ℓ^{-gamma}, then a propagator decay
#   G(ℓ) ~ ℓ^{-beta} with beta=(1-gamma)/2 is the boundary that keeps prices
#   diffusive (no superdiffusion).
# -----------------------------------------------------------------------------
def beta_from_gamma(gamma):
    """Return exponent beta for the (integrated) propagator decay 
        G(lag) = lag**-beta 
    that compensates a sign-autocorrelation 
        C(lag) = lag**-gamma.
    """
    return (1-gamma)/2.
    
# -----------------------------------------------------------------------------
# G_pow(steps, beta)
# Purpose:
#   Construct a simple power-law integrated kernel G(ℓ) ~ ℓ^{-beta} for ℓ>=1,
#   with G(0)=0. Mainly used for synthetic tests / sanity checks.
# -----------------------------------------------------------------------------
def G_pow(steps, beta):
    """Return power-law Propagator kernel G(l). l=0...steps"""
    G = np.arange(1,steps)**-beta#+1
    G = np.r_[0, G]
    return G
    
# -----------------------------------------------------------------------------
# k_pow(steps, beta)
# Purpose:
#   Return the differential kernel g(ℓ) = G(ℓ+1)-G(ℓ) derived from G_pow().
# -----------------------------------------------------------------------------
def k_pow(steps, beta):
    """Return increment of power-law propagator kernel g. l=0...steps"""
    return np.diff(G_pow(steps, beta))
    
# TIM1 specific 
# =====================================================================

# -----------------------------------------------------------------------------
# calibrate_tim1(c, Sl, maxlag=1e4)
# Purpose:
#   Calibrate the TIM1 differential kernel g by solving a Toeplitz system
#   relating the sign autocovariance c(ℓ) and the measured differential response
#   S(ℓ).
#
# Method:
#   - extract positive lags of S
#   - solve_toeplitz(c[:maxlag], S_pos[:maxlag]) to obtain g
#
# Notes:
#   Conditioning can be fragile at large maxlag if c is noisy; this repo often
#   smooths the estimated kernel tail afterward.
# -----------------------------------------------------------------------------
def calibrate_tim1(c, Sl, maxlag=10**4):
    """Return empirical estimate TIM1 kernel
    
    Parameters:
    ===========
    
    c: array-like
        Cross-correlation (covariance).
    Sl: array-like
        Price-response. If the response is differential, so is the returned
        kernel.
    maxlag: int
        length of the kernel.
    See also: integrate, g2_empirical, tim1
 
    """
    lS = int(len(Sl) / 2)
    g = solve_toeplitz(c[:maxlag], Sl[lS:lS+maxlag])
    return g

# -----------------------------------------------------------------------------
# tim1(s, G, sfunc=np.sign)
# Purpose:
#   Simulate the Transient Impact Model with a single kernel:
#     - pass G to get "price-like" output
#     - pass g (differential kernel) to get "return-like" output
#   This is just propagate(...).
# -----------------------------------------------------------------------------
def tim1(s, G, sfunc=np.sign):
    """Simulate Transient Impact Model 1, return price or return.
    
    Result is the price p when the bare responses G is passed
    and the 1 step ahead return p(t+1)-p(t) for the differential kernel 
    g, where G == numpy.cumsum(g).
    
    Parameters:
    ===========
    
    s: array-like
        Order signs
    G: array-like
        Kernel
    
    See also: calibrate_tim1, integrate, tim2, hdim2.
    """
    return propagate(s, G, sfunc=sfunc)

# TIM2 specific
# =====================================================================

# -----------------------------------------------------------------------------
# calibrate_tim2(nncorr, cccorr, cncorr, nccorr, Sln, Slc, maxlag)
# Purpose:
#   Calibrate the two-kernel TIM2 (non-price-changing vs price-changing trades).
#
# Idea:
#   Build a block correlation matrix from the 2-point covariances and solve:
#       [C_nn  C_cn] [g_n] = [S_n]
#       [C_nc  C_cc] [g_c]   [S_c]
#   where each block is Toeplitz (constructed by corr_mat).
#
# Output:
#   (g_n, g_c) differential kernels for n- and c-events.
# -----------------------------------------------------------------------------
def calibrate_tim2(
        nncorr, cccorr, cncorr, nccorr, Sln, Slc, maxlag=2**10
    ):
    """
    Return empirical estimate for both kernels of the TIM2.
    (Transient Impact Model with two propagators)
    
    Parameters:
    ===========
    
    nncorr: array-like
        Cross-covariance between non-price-changing (n-) orders.
    cccorr: array-like
        Cross-covariance between price-changing (c-) orders.
    cncorr: array-like
        Cross-covariance between c- and n-orders
    nccorr: array-like
        Cross-covariance between n- and c-orders.
    Sln: array-like
        (incremental) price response for n-orders
    Slc: array-like
        (incremental) price response for c-orders
    maxlag: int
        Length of the kernels.
    
    See also: calibrate_tim1, calibrate_hdim2
    """
    # incremental response
    lSn = int(len(Sln) / 2)
    lSc = int(len(Slc) / 2)
    S = np.concatenate([Sln[lSn:lSn+maxlag], Slc[lSc:lSc+maxlag]])
    
    # covariance matrix
    mat_fn = lambda x: corr_mat(x, maxlag=maxlag)
    C = np.bmat([
        [mat_fn(nncorr), mat_fn(cncorr)], 
        [mat_fn(nccorr), mat_fn(cccorr)]
    ])
        
    # solve
    g = solve(C, S)
    gn = g[:maxlag]
    gc = g[maxlag:]
            
    return gn, gc

# -----------------------------------------------------------------------------
# tim2(s, c, G_n, G_c, sfunc=np.sign)
# Purpose:
#   Simulate TIM2 by splitting the sign stream into two event types:
#     - c == True  : price-changing events use kernel G_c
#     - c == False : non-price-changing events use kernel G_n
#
# Notes:
#   - `c` is required to be boolean (asserted).
#   - In the papers, this split is crucial for large-tick assets where many
#     trades do not move the midprice.
# -----------------------------------------------------------------------------
def tim2(s, c, G_n, G_c, sfunc=np.sign):
    """Simulate Transient Impact Model 2
    
    Returns prices when integrated kernels are passed as arguments
    or returns for differential kernels.
    
    Parameters:
    ===========
    s: array
        Trade signs
    c: array
        Trade labels (1 = change; 0 = no change)
    G_n: array
        Kernel for non-price-changing trades
    G_c: array
        Kernel for price-changing trades
    sfunc: function [optional]
        Function to apply to signs. Default: np.sign.
        
    See also: calibrate_tim2, tim1, hdim2.
    """
    assert c.dtype == bool, "c must be a boolean indicator!"
    return propagate(s * c, G_c) + propagate(s * (~c), G_n)

    
# HDIM2 specific
# =====================================================================

# -----------------------------------------------------------------------------
# calibrate_hdim2(Cnnc, Cccc, Ccnc, Sln, Slc, maxlag=None, force_lag_zero=True)
# Purpose:
#   Calibrate HDIM2 (History Dependent Impact Model with two propagators).
#
# Inputs are *matrices* of 3-point correlations:
#   Cnnc(ℓ1,ℓ2), Cccc(ℓ1,ℓ2), Ccnc(ℓ1,ℓ2)
# corresponding to correlations between an unlagged label (change indicator)
# and two lagged signed-event streams (see scorr.x3corr / Felix paper).
#
# Method:
#   Build a 2x2 block matrix from these correlation matrices and solve for
#   (k_n, k_c) such that model responses match measured S_n, S_c.
#
# force_lag_zero:
#   Applies a constraint to pin down the system at lag 0 (helps conditioning).
# -----------------------------------------------------------------------------
def calibrate_hdim2(
        Cnnc, Cccc, Ccnc, Sln, Slc,
        maxlag=None, force_lag_zero=True
    ):
    """Return empirical estimate for both kernels of the HDIM2.
    (History Dependent Impact Model with two propagators).
    
    Requres three-point correlation matrices between the signs of one 
    non-lagged and two differently lagged orders.
    We distinguish between price-changing (p-) and non-price-changing (n-)
    orders. The argument names corresponds to the argument order in 
    spectral.x3corr.
    
    Parameters:
    ===========
    Cnnc: 2d-array-like
        Cross-covariance matrix for n-, n-, c- orders.
    Cccc: 2d-array-like
        Cross-covariance matrix for c-, c-, c- orders.
    Ccnc: 2d-array-like
        Cross-covariance matrix for c-, n-, c- orders.
    Sln: array-like
        (incremental) lagged price response for n-orders
    Slc: array-like
        (incremental) lagged price response for c-orders
    maxlag: int
        Length of the kernels.
        
    See also: hdim2,
    """
    maxlag = maxlag or min(len(Cccc), len(Sln))/2
    
    # incremental response
    lSn = int(len(Sln) / 2)
    lSc = int(len(Slc) / 2)
    S = np.concatenate([
        Sln[lSn:lSn+maxlag], 
        Slc[lSc:lSc+maxlag]
    ])
    
    # covariance matrix
    Cncc = Ccnc.T
    C = np.bmat([
        [Cnnc[:maxlag,:maxlag], Ccnc[:maxlag,:maxlag]], 
        [Cncc[:maxlag,:maxlag], Cccc[:maxlag,:maxlag]]
    ])
    
    if force_lag_zero:
        C[0,0] = 1
        C[0,1:] = 0
    
    # solve
    g = solve(C, S)
    gn = g[:maxlag]
    gc = g[maxlag:]
    
    return gn, gc
    
# -----------------------------------------------------------------------------
# hdim2(s, c, k_n, k_c, sfunc=np.sign)
# Purpose:
#   Simulate HDIM2. Compared to TIM2, HDIM2 enforces *label consistency*:
#   when the event is labeled as non-price-changing (c=False), the model return
#   is forced to be exactly zero via multiplication by `c`.
#
# This directly addresses the "TIM2 inconsistency" discussed in Felix:
# TIM2 can (in calibration) assign non-zero immediate impact to non-changing
# events, while HDIM2 cannot by construction.
# -----------------------------------------------------------------------------
def hdim2(s, c, k_n, k_c, sfunc=np.sign):
    """Simulate History Dependent Impact Model 2, return return.
    
    Parameters:
    ===========
    s: array
        Trade signs
    c: array
        Trade labels (1 = change; 0 = no change)
    k_n: array
        Differential kernel for non-price-changing trades
    k_c: array
        Differential kernel for price-changing trades
    sfunc: function [optional]
        Function to apply to signs. Default: np.sign.
        
    See also: calibrate_hdim2, tim2, tim1.
    """
    assert c.dtype == bool, "c must be a boolean indicator!"
    return c * (propagate(s * c, k_c) + propagate(s * (~c), k_n))
