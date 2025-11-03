import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pykalman import KalmanFilter
from hmmlearn.hmm import GaussianHMM
import vectorbt as vbt


tickers = ['MA', 'V'] 
data = yf.download(tickers, start="2020-01-01", end="2025-01-01", progress=False, auto_adjust=False)['Adj Close'] 
df = pd.DataFrame(data)

entry_t = 0.6
exit_t  = 0.5

# 1. Kalman filter
def stima_QR_EM(df_train, em_iter=50): # Estimate Q and R
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

    Q_hat = kf.transition_covariance.copy()
    R_hat = kf.observation_covariance.copy()
    return Q_hat, R_hat

N_train = 315
Q_hat, R_hat = stima_QR_EM(df.iloc[:N_train])


def kalman_spread(df, Q, R):        
    n_obs = len(df)
    y_vals = df.iloc[:,0].values
    x_vals = df.iloc[:,1].values 

    A = np.eye(2)                                    # Transition Matrix  

    # Observation matrices pre-computate
    observation_matrices = np.zeros((n_obs, 1, 2))
    observation_matrices[:, 0, 0] = 1.0
    observation_matrices[:, 0, 1] = x_vals

    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=observation_matrices,  
        initial_state_mean=[0.0, 1.0],
        initial_state_covariance=np.eye(2) * 100,     
        transition_covariance = Q,                    # rumore di processo
        observation_covariance = R                    # rumore di osservazione
    )

    state_means, _ = kf.filter(y_vals.reshape(-1, 1))  
    alpha_t = state_means[:, 0]
    beta_t  = state_means[:, 1]
    spread  = y_vals - (alpha_t + beta_t * x_vals)
    return alpha_t, beta_t, spread




alpha_t, beta_t, spread_kf = kalman_spread(df, Q=Q_hat, R=R_hat)
df['alpha_t']   = alpha_t
df['beta_t']    = beta_t
df['spread_kf'] = spread_kf


# 2. Spread Standardization
window_size = 60 # half life x3
spread_kf = df['spread_kf']
rm = spread_kf.rolling(window_size, min_periods=1).mean().shift(1)
rs = spread_kf.rolling(window_size, min_periods=1).std().shift(1)
zscore_kf = (spread_kf - rm) / rs
zscore_kf = zscore_kf.replace([np.inf, -np.inf], 0).fillna(0)
zscore_kf = zscore_kf.clip(-8, 8)

df['rolling_mean'] = rm
df['rolling_std'] = rs
df['z_kf'] = zscore_kf
df.drop(columns=['spread_kf'], inplace=True)


# 3. Hidden Markov Model
window = 315
zs = df['z_kf'].values.reshape(-1, 1)   # (T, 1)
n = len(zs)

model = GaussianHMM(
    n_components=2,
    covariance_type='diag',
    n_iter=200,
    tol=1e-4,   
    init_params="stmc"
)
model.fit(zs[:window]) 

state_vars = model.covars_.reshape(model.n_components, -1).sum(axis=1)
hi_var_state = int(np.argmax(state_vars))
def remap(state_arr):
    # 1 = high variance, 0 = low variance
    return np.where(state_arr == hi_var_state, 1, 0)

regimes = np.empty(n - window, dtype=int)

for i in range(n - window):
    # Sequence till today
    seq = zs[i : i + window + 1]        
    states = model.predict(seq)           # Viterbi 
    regimes[i] = remap(states)[-1]        # last state = today state

df = df.iloc[window:].copy()
df['regime'] = regimes



# 4. Signals generation
def generate_signals(df, entry_thresh, exit_thresh, HMM=True):
    sig = np.zeros(len(df), dtype=int)
    position = 0

    for i in range(len(df)):
        z = df.loc[df.index[i], 'z_kf']
        regime = df.loc[df.index[i], 'regime']
        
        if HMM: # ACCESO
            if regime == 1: 
                position = 0           
            else:
                if position == 0:
                    # se non ho posizione, cerco quando entrare
                    if z >= entry_thresh:
                        position = -1  # spread alto → short 
                    elif z <= -entry_thresh:
                        position = 1   # spread basso → long 
                else:
                    # se ho già posizione, cerco quando uscire
                    if abs(z) < exit_thresh:
                        position = 0   # zscore ha ritracciato → chiudo
        else:   # SPENTO
                if position == 0:
                    if z >= entry_thresh:
                        position = -1  
                    elif z <= -entry_thresh:
                        position = 1   
                else:
                    if abs(z) < exit_thresh:
                        position = 0   
                    
        sig[i] = position

    return sig

df['signal'] = generate_signals(df, entry_thresh=entry_t, exit_thresh=exit_t, HMM=True) # already shifted by VBT




# 5. Backtest 

# Prices
y_ticker, x_ticker = tickers[:2]                
prices = df[[y_ticker, x_ticker]].astype(float)

# Weights
beta_lag = df['beta_t'].shift(1).ffill().clip(-10, 10)
pos = df['signal'].fillna(0.0) # vbt fa già lo shift(1) 
gross = 1.0 + beta_lag.abs()

w_y =  pos / gross                 
w_x = -pos * beta_lag / gross      

weights = pd.DataFrame({
    y_ticker: w_y,
    x_ticker: w_x
}, index=prices.index).fillna(0.0)


# BT
pf = vbt.Portfolio.from_orders(
    close=prices,
    size=weights,
    size_type='targetpercent',
    init_cash=10_000,
    fees=0.0001,
    cash_sharing=True,
    freq='1D'
)

stats = pf.stats()
wanted = [
    'End Value',
    'Total Return [%]',
    'Max Drawdown [%]',
    'Max Drawdown Duration',
    'Total Trades',
    'Win Rate [%]',
    'Sharpe Ratio',
    'Calmar Ratio',
    'Sortino Ratio',
]
subset = [m for m in wanted if m in stats.index]
print(stats.loc[subset].to_string())
#print(pf.stats()) # complete stats
#pf.plot().show()
