import os
import numpy as np
import CoolProp.CoolProp as CP
import xgboost as xgb

from Mathieu import Fluid_properties, Interface_resistance

'''
XGBoost surrogate (dtrfree) of the Mathieu TCL model, loaded from the native JSON files.
Inputs in SI: P_sat [Pa], DT [K], delta [m], theta_mic [deg], f accommodation coefficient.
DT, delta, theta_mic and f can be arrays (broadcast together); fluid and P_sat are scalars.
Outputs are arrays: q_p [W/m] and theta_app [deg].

Predictors (log-transformed, in this order):
    Pi2 = sigma**2/(h_lv*mu_l**2)
    Pi3 = delta/(k_l*R_i)                  #not the Pi3 of MathieuModel.get_pi3
    Pi4 = rho_l*k_l*R_i/mu_l*h_lv**0.5
    Pi5 = k_l*DT/(h_lv*mu_l)
    Pi6 = theta_mic                        #deg, not theta_app/theta_mic
Targets: Pi1 = q_p/(h_lv*mu_l) and theta_app.
'''

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ['pi1', 'theta_app']
MODELS = {}

#Trained range of the dataset (Water and O2 were not used in the fit, Ne was dropped)
RANGE = {'P [bar]': [0.6, 154], 'DT [K]': [0.1, 40], 'delta [um]': [0.5, 50],
         'theta_mic [deg]': [5, 40], 'f': [0.0198, 1],
         'pi2': [2.2e-4, 0.075], 'pi3': [0.115, 3.9e4], 'pi4': [5.96, 2.2e4], 'pi5': [4.1e-4, 0.40]}


def load_models():
    if not MODELS:
        for target in TARGETS:
            MODELS[target] = xgb.Booster()
            MODELS[target].load_model(os.path.join(HERE, f'ml_dtrfree_xgb_{target}.json'))

    return MODELS


def Range_check(fluid, P_sat, DT, delta, theta_mic, f, pi2, pi3, pi4, pi5):
    values = {'P [bar]': P_sat/1e5, 'DT [K]': DT, 'delta [um]': delta*1e6, 'theta_mic [deg]': theta_mic,
              'f': f, 'pi2': pi2, 'pi3': pi3, 'pi4': pi4, 'pi5': pi5}

    for name, v in values.items():
        [lo, hi] = RANGE[name]
        v = np.atleast_1d(v)
        n_out = ((v<lo) | (v>hi)).sum()
        if n_out>0:
            print(f"Warning: {name} is outside the trained range {lo}-{hi} in {n_out}/{v.size} points (min {v.min():.4g}, max {v.max():.4g})")

    if fluid != 'FC72' and P_sat>0.7*CP.PropsSI('Pcrit', fluid):
        print("Warning: P_sat is above 0.7*P_crit, the maximum used in the dataset.")


def compute_pi_groups(fluid, P_sat, DT, delta, theta_mic, f=1.0):
    [R, T_sat, rho_l, rho_v, h_lv, k_l, mu_l, sigma] = Fluid_properties(P_sat, fluid)

    DT, delta, theta_mic, f = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, dtype=np.float64))
                                                    for v in [DT, delta, theta_mic, f]])

    R_i = Interface_resistance(R, T_sat, h_lv, rho_v, f)

    pi2 = np.full(DT.shape, sigma**2/h_lv/mu_l**2)
    pi3 = delta/k_l/R_i
    pi4 = rho_l*k_l*R_i/mu_l*h_lv**0.5
    pi5 = k_l*DT/h_lv/mu_l
    pi6 = theta_mic

    Range_check(fluid, P_sat, DT, delta, theta_mic, f, pi2, pi3, pi4, pi5)

    return [pi2, pi3, pi4, pi5, pi6]


def predict_pi1_theta(pi2, pi3, pi4, pi5, pi6):
    models = load_models()

    cols = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, dtype=np.float64))
                                 for v in [pi2, pi3, pi4, pi5, pi6]])
    Xlog = np.log(np.column_stack(cols)).astype(np.float32)
    DX = xgb.DMatrix(Xlog)

    pi1 = np.exp(models['pi1'].predict(DX).astype(np.float64))
    theta_app = np.exp(models['theta_app'].predict(DX).astype(np.float64))

    return [pi1, theta_app]


def qprime_from_pi1(pi1, fluid, P_sat):
    [R, T_sat, rho_l, rho_v, h_lv, k_l, mu_l, sigma] = Fluid_properties(P_sat, fluid)
    q_p = pi1*h_lv*mu_l   #W/m

    return q_p


def predict_q_theta(fluid, P_sat, DT, delta, theta_mic=8, f=1.0):
    [pi2, pi3, pi4, pi5, pi6] = compute_pi_groups(fluid, P_sat, DT, delta, theta_mic, f)
    [pi1, theta_app] = predict_pi1_theta(pi2, pi3, pi4, pi5, pi6)
    q_p = qprime_from_pi1(pi1, fluid, P_sat)

    return [q_p, theta_app]


if __name__ == '__main__':
    [q_p, theta_app] = predict_q_theta('Water', 1e5, 10, 20e-6, 8)
    print("q_p [W/m] ", q_p, "theta_app [deg] ", theta_app)
