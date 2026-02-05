import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pykalman import KalmanFilter
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm, skew, kurtosis
import vectorbt as vbt


# =============================================================================
# CONFIGURAZIONE
# =============================================================================
TICKERS = ['MA', 'V']
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

# Parametri strategia
ENTRY_THRESHOLD = 0.6
EXIT_THRESHOLD = 0.4
HMM_ENABLED = True

# Parametri modello
N_TRAIN = 315
WINDOW_SIZE = 60
HMM_WINDOW = 315
EM_ITERATIONS = 50

# Parametri backtest
INITIAL_CASH = 10_000
FEES = 0.0001
N_TRIALS = 60  # Numero varianti testate


# =============================================================================
# 1. DATI
# =============================================================================
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, 
                   progress=False, auto_adjust=False)['Adj Close']
df = pd.DataFrame(data)


# =============================================================================
# 2. KALMAN FILTER
# =============================================================================
def estimate_kalman_params(df_train, em_iter=50):
    """Stima Q e R tramite EM algorithm."""
    y = df_train.iloc[:, 0].values.reshape(-1, 1)
    x = df_train.iloc[:, 1].values
    n_obs = len(df_train)

    A = np.eye(2)
    H = np.zeros((n_obs, 1, 2))
    H[:, 0, 0] = 1.0
    H[:, 0, 1] = x

    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        initial_state_mean=[0.0, 1.0],
        initial_state_covariance=np.eye(2) * 100
    ).em(y, n_iter=em_iter, em_vars=['transition_covariance', 'observation_covariance'])

    return kf.transition_covariance.copy(), kf.observation_covariance.copy()


def compute_kalman_spread(df, Q, R):
    """Calcolo spread."""
    n_obs = len(df)
    y_vals = df.iloc[:, 0].values
    x_vals = df.iloc[:, 1].values

    A = np.eye(2)
    observation_matrices = np.zeros((n_obs, 1, 2))
    observation_matrices[:, 0, 0] = 1.0
    observation_matrices[:, 0, 1] = x_vals

    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=observation_matrices,
        initial_state_mean=[0.0, 1.0],
        initial_state_covariance=np.eye(2) * 100,
        transition_covariance=Q,
        observation_covariance=R
    )

    state_means, _ = kf.filter(y_vals.reshape(-1, 1))
    alpha_t = state_means[:, 0]
    beta_t = state_means[:, 1]
    spread = y_vals - (alpha_t + beta_t * x_vals)
    
    return alpha_t, beta_t, spread


# Stima parametri e calcola spread
Q_hat, R_hat = estimate_kalman_params(df.iloc[:N_TRAIN], em_iter=EM_ITERATIONS)
alpha_t, beta_t, spread_kf = compute_kalman_spread(df, Q=Q_hat, R=R_hat)

df['alpha_t'] = alpha_t
df['beta_t'] = beta_t
df['spread_kf'] = spread_kf


# =============================================================================
# 3. STANDARDIZZAZIONE SPREAD (Z-SCORE)
# =============================================================================
def compute_zscore(spread, window_size):
    """Calcola z-score rolling dello spread."""
    rm = spread.rolling(window_size, min_periods=1).mean().shift(1)
    rs = spread.rolling(window_size, min_periods=1).std().shift(1)
    zscore = (spread - rm) / rs
    zscore = zscore.replace([np.inf, -np.inf], 0).fillna(0)
    zscore = zscore.clip(-8, 8)
    return zscore, rm, rs


zscore_kf, rolling_mean, rolling_std = compute_zscore(df['spread_kf'], WINDOW_SIZE)
df['rolling_mean'] = rolling_mean
df['rolling_std'] = rolling_std
df['z_kf'] = zscore_kf
df.drop(columns=['spread_kf'], inplace=True)


# =============================================================================
# 4. HIDDEN MARKOV MODEL
# =============================================================================
def train_hmm(z_scores, window):
    """Addestramento HMM."""
    model = GaussianHMM(
        n_components=2,
        covariance_type='diag',
        n_iter=200,
        tol=1e-4,
        random_state=42,
        init_params="stmc"
    )
    model.fit(z_scores[:window])
    
    # Identifica stato ad alta varianza
    state_vars = model.covars_.reshape(model.n_components, -1).sum(axis=1)
    hi_var_state = int(np.argmax(state_vars))
    
    return model, hi_var_state


def predict_regimes(z_scores, model, hi_var_state, window):
    """Predice regimi usando rolling Viterbi."""
    n = len(z_scores)
    regimes = np.empty(n - window, dtype=int)
    
    for i in range(n - window):
        seq = z_scores[i : i + window + 1]
        states = model.predict(seq)
        # Rimappa: 1 = alta varianza, 0 = bassa varianza
        current_state = states[-1]
        regimes[i] = 1 if current_state == hi_var_state else 0
    
    return regimes


# Addestra HMM e predici regimi
zs = df['z_kf'].values.reshape(-1, 1)
hmm_model, hi_var_state = train_hmm(zs, HMM_WINDOW)
regimes = predict_regimes(zs, hmm_model, hi_var_state, HMM_WINDOW)

df = df.iloc[HMM_WINDOW:].copy()
df['regime'] = regimes


# =============================================================================
# 5. GENERAZIONE SEGNALI
# =============================================================================
def generate_signals(df, entry_thresh, exit_thresh, use_hmm=True):
    """
    Genera segnali di trading.
    
    Returns:
        1: long spread (buy Y, sell X)
        -1: short spread (sell Y, buy X)
        0: nessuna posizione
    """
    sig = np.zeros(len(df), dtype=int)
    position = 0

    for i in range(len(df)):
        z = df.loc[df.index[i], 'z_kf']
        regime = df.loc[df.index[i], 'regime']
        
        # Se HMM attivo e siamo in regime ad alta varianza, chiudi posizioni
        if use_hmm and regime == 1:
            position = 0
        else:
            if position == 0:  # Cerca entry
                if z >= entry_thresh:
                    position = -1  # spread alto → short
                elif z <= -entry_thresh:
                    position = 1   # spread basso → long
            else:  # Cerca exit
                if abs(z) < exit_thresh:
                    position = 0
        
        sig[i] = position

    return sig


df['signal'] = generate_signals(df, ENTRY_THRESHOLD, EXIT_THRESHOLD, HMM_ENABLED)


# =============================================================================
# 6. METRICHE STATISTICHE
# =============================================================================
def sharpe_per_period(returns):
    """Sharpe ratio non annualizzato (per-period)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    mu = r.mean()
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else mu / sd


def sharpe_pvalue_two_sided(returns, sr0=0.0):
    """
    p-value (two-sided) per test H0: SR <= sr0.
    Usa approssimazione normale del t-statistic.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 2:
        return np.nan
    
    sr_hat = sharpe_per_period(r)
    z = (sr_hat - sr0) * np.sqrt(T - 1)
    return 2.0 * (1.0 - norm.cdf(abs(z)))


def bonferroni_correction(pvals):
    """Bonferroni correction per FWER control: p_adj = min(p * m, 1)."""
    p = np.asarray(pvals, dtype=float)
    m = np.sum(np.isfinite(p))
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    out[mask] = np.minimum(p[mask] * m, 1.0)
    return out


def deflated_sharpe_ratio(returns, N_trials, sr_trials=None):
    """
    Deflated Sharpe Ratio (Bailey & López de Prado).
    
    Args:
        returns: returns per-period della strategia selezionata
        N_trials: numero di strategie testate
        sr_trials: Sharpe di tutti i trial (opzionale)
    
    Returns:
        Probabilità che SR sia statisticamente significativo dopo data snooping
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 2 or N_trials < 1:
        return np.nan

    sr_star = sharpe_per_period(r)

    # Momenti campionari
    g3 = skew(r, bias=False) if T >= 3 else 0.0
    g4 = kurtosis(r, fisher=False, bias=False) if T >= 4 else 3.0

    # Varianza dello Sharpe ratio
    if sr_trials is not None:
        srt = np.asarray(sr_trials, dtype=float)
        srt = srt[np.isfinite(srt)]
        V_sr = np.var(srt, ddof=1) if srt.size >= 2 else 1.0 / (T - 1)
    else:
        V_sr = 1.0 / (T - 1)

    # Soglia SR0 = expected max Sharpe di N_trials sotto H0
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    z1 = norm.ppf(1.0 - 1.0 / N_trials)
    z2 = norm.ppf(1.0 - 1.0 / (N_trials * np.e))
    sr0 = np.sqrt(V_sr) * ((1.0 - gamma) * z1 + gamma * z2)

    # DSR
    denom = np.sqrt(1.0 - g3 * sr0 + ((g4 - 1.0) / 4.0) * (sr0 ** 2))
    z = ((sr_star - sr0) * np.sqrt(T - 1.0)) / denom
    
    return float(norm.cdf(z))


# =============================================================================
# 7. BACKTEST
# =============================================================================
def prepare_portfolio_weights(df, tickers):
    """Calcola i pesi del portfolio per il backtest."""
    beta_lag = df['beta_t'].shift(1).ffill().clip(-10, 10)
    pos = df['signal'].fillna(0.0)
    gross = 1.0 + beta_lag.abs()

    y_ticker, x_ticker = tickers[:2]
    w_y = pos / gross
    w_x = -pos * beta_lag / gross

    weights = pd.DataFrame({
        y_ticker: w_y,
        x_ticker: w_x
    }, index=df.index).fillna(0.0)
    
    return weights


def run_backtest(prices, weights, init_cash, fees):
    """Esegue backtest con VectorBT."""
    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type='targetpercent',
        init_cash=init_cash,
        fees=fees,
        cash_sharing=True,
        freq='1D'
    )
    return pf


def analyze_backtest_results(pf, N_trials):
    """Analizza risultati del backtest con metriche robuste."""
    # Returns giornalieri
    rets = pf.returns().dropna().values
    
    # Metriche base
    sr_hat = sharpe_per_period(rets)
    pval = sharpe_pvalue_two_sided(rets, sr0=0.0)
    pval_bonf = bonferroni_correction([pval])[0]
    dsr = deflated_sharpe_ratio(rets, N_trials=N_trials)
    
    # Print risultati
    print("=" * 70)
    print("STATISTICHE BACKTEST")
    print("=" * 70)
    print(pf.stats())
    print("\n" + "=" * 70)
    print("METRICHE ROBUSTE (Data Snooping Correction)")
    print("=" * 70)
    print(f"Sharpe Ratio (per-period):           {sr_hat:.4f}")
    print(f"p-value Sharpe (two-sided):          {pval:.4g}")
    print(f"p-value Bonferroni-corrected:        {pval_bonf:.4g}")
    print(f"Deflated Sharpe Ratio (prob):        {dsr:.4f}")
    print(f"N° trials considerati:               {N_trials}")
    print("=" * 70)
    
    # Interpretazione
    if pval_bonf < 0.05:
        print("✓ Strategia statisticamente significativa (α=0.05, Bonferroni)")
    else:
        print("✗ Strategia NON significativa dopo correzione Bonferroni")
    
    if dsr > 0.95:
        print("✓ DSR elevato: risultati robusti al data snooping")
    elif dsr > 0.5:
        print("⚠ DSR moderato: possibile overfitting")
    else:
        print("✗ DSR basso: probabile data snooping")
    print("=" * 70)
    
    return rets, sr_hat, pval, pval_bonf, dsr



# Esecuzione backtest
prices = df[TICKERS].astype(float)
weights = prepare_portfolio_weights(df, TICKERS)
pf = run_backtest(prices, weights, INITIAL_CASH, FEES)

results = analyze_backtest_results(pf, N_TRIALS)

pf.plot().show()
