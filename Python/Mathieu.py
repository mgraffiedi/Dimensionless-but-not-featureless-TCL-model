import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import warnings
from scipy.integrate import solve_ivp
from scipy.optimize import root

import FC72_lib as FC72


def Fluid_properties(P_sat, substance):
    if substance == 'FC72':
        [R, T_sat, rho_l, rho_v, h_lv, k_l, mu_l, sigma] = FC72.FC72_Properties(P_sat)
    else:
        R = CP.PropsSI('gas_constant',substance) / CP.PropsSI('molar_mass',substance)   #J/kg/K
        T_sat = CP.PropsSI('T','P',P_sat,'Q',0,substance)
        rho_l = CP.PropsSI('D','T',T_sat,'Q',0,substance)
        rho_v = CP.PropsSI('D','T',T_sat,'Q',1,substance)
        h_lv = CP.PropsSI('HMASS','T',T_sat,'Q',1,substance) - CP.PropsSI('HMASS','T',T_sat,'Q',0,substance)
        k_l = CP.PropsSI('CONDUCTIVITY','T',T_sat,'Q',0,substance)
        mu_l = CP.PropsSI('V','T',T_sat,'Q',0,substance)
        sigma = CP.PropsSI('I','T',T_sat,'Q',0,substance)

    return [R, T_sat, rho_l, rho_v, h_lv, k_l, mu_l, sigma]


def Interface_resistance(R, T_sat, h_lv, rho_v, f=1):
    R_i = T_sat * (2 * np.pi * R * T_sat)**0.5 / (h_lv**2 * rho_v) * (2 - f) / 2 / f

    return R_i


class MathieuModel:
    def __init__(self,P_sat,substance,theta_mic=8,f = 1,
                 maxx = None, maxy = 20e-6,
                 DP_shoot=None,l_sl = 1e-10,num_points = 1000):
        # Constants
        self.P_sat = P_sat  # Pa
        self.theta_mic = theta_mic

        [self.R, self.T_sat, self.rho_l, self.rho_v,
         self.h_lv, self.k_l, self.mu_l, self.sigma] = Fluid_properties(P_sat, substance)

        self.l_sl = l_sl
        self.f = f
        self.R_i = Interface_resistance(self.R, self.T_sat, self.h_lv, self.rho_v, self.f)
        if DP_shoot is None:
            DP_shoot = [1.6e6, 1e6, 0.4e6, 2e6, 5e6, 0.1e6, 0.01e6, 1e5, 1e4]
        self.DP_shoot = DP_shoot
        self.v_s = 0
        self.maxy = maxy  # m, if None, height stopping is disabled
        if maxx is None:
            maxx = 10*maxy
        self.maxx = maxx  # m
        self.Ds = maxx*3
        self.num_points = num_points

    def U_b_def(self,U_b=0):
        self.v_s = U_b

    def DT_def(self,DT=10):
        self.DT = float(DT)
        self.T_W = self.T_sat+self.DT

    def diff_eq(self, t, x):
        [axe, d, theta, DP, q_p] = x

        r = (d + self.l_sl) / np.sin(theta)   #Checked
        f_V = 8 / (2 * theta * np.cos(2 * theta) - np.sin(2 * theta))  #Checked
        f_v = 4 * np.sin(theta) / (2 * theta - np.sin(2 * theta))

        dx_ds = np.cos(theta)
        dd_ds = np.sin(theta)
        dtheta_ds = max(1 / self.sigma * (DP),0)
        dDP_ds = min(-self.mu_l * (-q_p / self.rho_l / self.h_lv / r**3 * f_V + self.v_s * 1 / r**2 * f_v),0)
        dq_ds = max((self.T_W - self.T_sat * (1 + DP / self.h_lv / self.rho_l)) / (r * theta / self.k_l + self.R_i), 0)

        return [dx_ds, dd_ds, dtheta_ds, dDP_ds, dq_ds]

    def event_theta(self, t, y):
        return y[2] - np.pi

    event_theta.terminal  = True
    event_theta.direction = 1

    def event_maxy(self, t, y):
        return y[1] - self.maxy  # fires when δ(s) == maxy

    event_maxy.terminal = True
    event_maxy.direction = 1

    def set_Ds(self,Ds):
        self.Ds = Ds

    def initialize(self):
        self.s_span = (0, self.Ds)
        self.t_eval = np.linspace(self.s_span[0], self.s_span[1], self.num_points)

    def endP(self, x):
        [s, x_arr, y] = self.Param(x, coarse=True)  # no t_eval → only final point
        return y[3][-1]

    def simulate(self, verbose=False):
        self.theta = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for n_Ds in range(5):
                self.initialize()
                converged = False
                for DP_shoot in self.DP_shoot:
                    x0 = root(self.endP, [DP_shoot], method='hybr')

                    if abs(x0.fun) < 1:
                        converged = True
                        if verbose: print("DP_shoot", DP_shoot, "did converge, final DP_shoot", x0.x)
                        break
                    if verbose: print("DP_shoot", DP_shoot, "did not converge")

                if not converged:
                    break

                [s, x, y] = self.Param(x0.x)

                # x is checked only if the maxy event did not fire
                if self._stopped_by_maxy or x[-1] > self.maxx:
                    self.theta = y[2][-1]
                    return [s, x, y]

                self.set_Ds(self.Ds + (self.maxx - x[-1]) / np.cos(y[2][-1]) * 2)
                if verbose: print("x is short, new Ds", self.Ds)

        print(f"Warning: not converged (P_sat = {self.P_sat}, DT = {self.DT})")
        return [np.array([np.nan]), np.array([np.nan]), np.full((5,1), np.nan)]

    def Param(self, x, coarse=False):
        DP_shoot = x[0]
        y0 = np.array([0, 0, self.theta_mic/180*np.pi, DP_shoot, 0.0])

        events = [self.event_theta]
        if not coarse and self.maxy is not None:
            events.append(self.event_maxy)

        sol = solve_ivp(
            self.diff_eq,
            self.s_span,
            y0,
            method='Radau',
            events=events,
            t_eval=None if coarse else self.t_eval,
            rtol=1e-6,
            atol=1e-9,
        )
        s = sol.t
        y = sol.y
        # flag whether maxy event caused termination
        self._stopped_by_maxy = (
            self.maxy is not None and
            not coarse and
            len(sol.t_events[1]) > 0  # t_events[1] = maxy event
        )
        if self._stopped_by_maxy:
            # t_eval stops before the event, so the exact δ = maxy point is appended
            s = np.append(s, sol.t_events[1][0])
            y = np.column_stack([y, sol.y_events[1][0]])
        return [s, y[0], y]

    def plot(self, s, x, y):
        variable_names = [r"$\delta \, [\mu m]$", r"$\theta \, [^o]$",
                          r"$\Delta P \, [Pa]$", r"$q^{\prime} \, [W/m]$"]

        x_um = x*1e6
        y_plot = [y[1]*1e6, y[2]*180/np.pi, y[3], y[4]]

        # Create a 2x2 subplot
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        for i in range(4):
            ax = axs[i // 2, i % 2]  # Determine the subplot position (row, col)
            ax.plot(x_um, y_plot[i], label=variable_names[i])
            ax.set_xlabel(r'$x \, [\mu m]$')
            ax.set_ylabel(variable_names[i])
            ax.legend(frameon=False)
            ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()
