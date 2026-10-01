import numpy as np

M = 338e-3   #kg/mol
R_gas = 8.314371357587   #J/mol/K
R_rho = 8.314/338e-3   #J/kg/K, gas constant used for rho_v
T_crit = 176+273   #K
dT_fd = 0.1   #K, step of the forward difference for dP/dT


def Antoine_P(T):
    P_sat = 10**(9.729-1562/T)   #Pa

    return P_sat


def Antoine_T(P):
    T_sat = float(1562/(9.729-np.log10(P)))   #K

    return T_sat


def FC72_Properties(P_sat):
    if P_sat<0.6e5 or P_sat>13e5:
        print("Warning: P_sat is outside the range of the FC72 correlations (0.6-13 bar).")
        print("P_sat = ", P_sat)

    R = R_gas/M

    T_sat = Antoine_T(P_sat)

    rho_l = 1740-2.61*(T_sat-273.15)
    rho_v = Antoine_P(T_sat)/T_sat/R_rho

    dP_dT = (Antoine_P(T_sat+dT_fd)-Antoine_P(T_sat))/dT_fd
    h_lv = dP_dT*T_sat*(rho_v**-1-rho_l**-1)

    k_l = 0.06-0.00011*(T_sat-273.15)

    T_ln = np.log10(T_sat)
    nu_ln = -3.40700E+01*T_ln**3 + 2.58944E+02*T_ln**2 - 6.59140E+02*T_ln + 5.61286E+02
    nu_l = 10**-6*10**nu_ln
    mu_l = nu_l*rho_l

    sigma_0 = 10.4e-3/(1-293.15/T_crit)**1.28
    sigma = sigma_0*(1-T_sat/T_crit)**1.28

    return [R, T_sat, rho_l, rho_v, h_lv, k_l, mu_l, sigma]
