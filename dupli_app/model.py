import numpy as np
import pandas as pd
import pickle, os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

PRESS_SPEED  = 5217
WORKING_DAYS = 250
PROFIT_PER   = 0.0025
MODEL_PATH   = "gb_before.pkl"
FEATURES     = ['Qty', 'Setup_min', 'Log_Qty']

FLOOR_SETUP = {
    500:  1.47,
    750:  2.39,
    1000: 1.48,
    2000: 7.00,
    2500: 2.62,
    2685: 1.70,
    5000: 3.50,
    7500: 3.00,
    10000:7.45,
}

AFTER_PILOT_OBS = [
    {'Qty':750,  'Full_Cycle_min':8.11},
    {'Qty':2500, 'Full_Cycle_min':14.40},
    {'Qty':2685, 'Full_Cycle_min':15.00},
    {'Qty':2500, 'Full_Cycle_min':15.88},
    {'Qty':2500, 'Full_Cycle_min':14.90},
    {'Qty':1000, 'Full_Cycle_min':7.58},
    {'Qty':7500, 'Full_Cycle_min':9.00},
    {'Qty':2000, 'Full_Cycle_min':12.53},
    {'Qty':1000, 'Full_Cycle_min':5.00},
]
df_after_obs = pd.DataFrame(AFTER_PILOT_OBS)


def get_setup_time(qty: int) -> float:
    known = sorted(FLOOR_SETUP.keys())
    below = [q for q in known if q <= qty]
    above = [q for q in known if q >= qty]
    if below and above and below[-1] != above[0]:
        q1, q2 = below[-1], above[0]
        t = (qty - q1) / (q2 - q1)
        return round(FLOOR_SETUP[q1] + t*(FLOOR_SETUP[q2]-FLOOR_SETUP[q1]), 2)
    nearest = min(known, key=lambda q: abs(q-qty))
    return FLOOR_SETUP[nearest]


def get_after_cycle(qty: int) -> dict:
    df = df_after_obs.copy()
    df['dist'] = abs(df['Qty'] - qty)
    df_near = df[df['dist'] == df['dist'].min()]
    return {
        'cycle_min':    round(df_near['Full_Cycle_min'].mean(), 2),
        'n_obs':        int(len(df_near)),
        'nearest_qty':  int(df_near.iloc[0]['Qty']),
    }


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None


def predict_before(qty: int) -> dict:
    model = load_model()
    if model is None:
        return {'error': 'Model not found. Run train.py first.'}
    setup = get_setup_time(qty)
    log_q = np.log(max(1, qty))
    X     = pd.DataFrame([[qty, setup, log_q]], columns=FEATURES)
    cycle = float(model.predict(X)[0])
    return {
        'qty':       qty,
        'setup_min': setup,
        'cycle_min': round(cycle, 2),
    }


def calculate_impact(qty: int) -> dict:
    before       = predict_before(qty)
    after        = get_after_cycle(qty)
    before_cycle = before['cycle_min']
    after_cycle  = after['cycle_min']
    saving       = round(before_cycle - after_cycle, 2)
    pct          = round(saving / before_cycle * 100, 1) if before_cycle > 0 else 0
    jobs_per_day = 17.4
    time_freed   = saving * jobs_per_day / 60
    extra_env_day= time_freed * PRESS_SPEED
    return {
        'qty':               qty,
        'setup_min':         before['setup_min'],
        'before_cycle':      before_cycle,
        'after_cycle':       after_cycle,
        'after_nearest_qty': after['nearest_qty'],
        'after_n_obs':       after['n_obs'],
        'saving_per_job':    saving,
        'pct_improvement':   pct,
        'time_freed_hrs':    round(time_freed, 2),
        'extra_env_day':     int(extra_env_day),
        'extra_env_year':    int(extra_env_day * WORKING_DAYS),
        'extra_profit':      round(extra_env_day * WORKING_DAYS * PROFIT_PER, 2),
    }


def calc_assumption_gains(daily_input: int) -> dict:
    a1_gain = 6.5  * PRESS_SPEED
    a2_gain = 2.5  * 2 * PRESS_SPEED
    a3_gain = a2_gain + 2.0 * PRESS_SPEED
    return {
        'current':      daily_input,
        'a1':           int(daily_input + a1_gain),
        'a2':           int(daily_input + a2_gain),
        'a3':           int(daily_input + a3_gain),
        'a1_gain':      int(a1_gain),
        'a2_gain':      int(a2_gain),
        'a3_gain':      int(a3_gain),
        'a1_annual':    round((daily_input + a1_gain)*WORKING_DAYS/1e6, 2),
        'a2_annual':    round((daily_input + a2_gain)*WORKING_DAYS/1e6, 2),
        'a3_annual':    round((daily_input + a3_gain)*WORKING_DAYS/1e6, 2),
        'a1_profit':    round((daily_input + a1_gain)*WORKING_DAYS*PROFIT_PER, 0),
        'a2_profit':    round((daily_input + a2_gain)*WORKING_DAYS*PROFIT_PER, 0),
        'a3_profit':    round((daily_input + a3_gain)*WORKING_DAYS*PROFIT_PER, 0),
    }


REQUIRED_DAILY = {
    '15M (envelopes.com only)':     int(15_000_000 / WORKING_DAYS),
    '25M (combined target)':        int(25_000_000 / WORKING_DAYS),
    '35M (maximum goal)':           int(35_000_000 / WORKING_DAYS),
    '45M (full capacity)':          int(45_000_000 / WORKING_DAYS),
}
