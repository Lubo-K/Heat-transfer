# -*- coding: utf-8 -*-
"""
Program na výpočet ohrevu vodiča prúdom I ponoreného do LN2
Created on Fri Dec 27 15:40:46 2024
@author: Lubos
"""

from tkinter import Tk, filedialog, Label, Entry, Button, font
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import json
import os


class ParameterDialog:
    def __init__(self, root, params=None):
        self.dialog = Tk()
        self.dialog.title("Nastavenie parametrov")
        self.entries = {}

        # Load parameters from file if available
        if params is None:
            params = self.load_params()

        self.params = params

        # Vytvorenie formulára
        row = 0
        for label, default in self.params.items():
            tk_label = Label(self.dialog, text=label)
            tk_label.grid(row=row, column=0, sticky="w", padx=10, pady=5)

            entry = Entry(self.dialog)
            entry.insert(0, str(default))
            entry.grid(row=row, column=1, padx=10, pady=5)
            self.entries[label] = entry

            row += 1

        # Tlačidlo potvrdenia
        bold_font = font.Font(weight="bold")
        confirm_btn = Button(self.dialog, text="OK", font=bold_font,
                             command=self.on_confirm, width=8, height=1)
        confirm_btn.grid(row=row, column=0, columnspan=2, pady=10)

    def on_confirm(self):
        params = {}
        for label, entry in self.entries.items():
            params[label] = float(entry.get())
        self.params = params
        self.save_params()
        self.dialog.quit()
        self.dialog.destroy()

    def get_parameters(self):
        self.dialog.mainloop()
        return self.params

    def save_params(self):
        with open("params.json", "w") as file:
            json.dump(self.params, file)

    def load_params(self):
        if os.path.exists("params.json"):
            with open("params.json", "r") as file:
                return json.load(file)
        else:
            return {
                "Hmotnosť vodiča (m) [kg]": 0.0001995,
                "Teplota okolia (T_env) [K]": 77,
                "Počiatočná teplota vodiča (T_initial) [K]": 77,
                "Plocha vodi?a (A) [m^2]": 8.2693E-5,
                "Časový krok (dt) [s]": 0.001,
            }


# Výzva na nastavenie parametrov
Tk().withdraw()  # Skryje hlavné okno Tkinteru
param_dialog = ParameterDialog(None)
params = param_dialog.get_parameters()

# Parametre zo vstupného okna
m = params["Hmotnosť vodiča (m) [kg]"]
T_env = params["Teplota okolia (T_env) [K]"]
T_initial = params["Počiatočná teplota vodiča (T_initial) [K]"]
A = params["Plocha vodiča (A) [m^2]"]
dt = params["Časový krok (dt) [s]"]


# Výzva na výber cesty k dátovým súborom
Tk().withdraw()  # Skryje hlavné okno Tkinteru
file_path_I = filedialog.askopenfilename(title="Vyber súbor I(t)", filetypes=[("Textové súbory", "*.txt")])
file_name = os.path.basename(file_path_I)

# Na?ítanie dát zo súborov
data_R = np.loadtxt('Copper_R(T).txt', delimiter=';')
T_table_R = data_R[:, 0]  # teploty (K) pre R
R_table = data_R[:, 1]  # odpory (Ohm)

data_c = np.loadtxt('Coeff_Cp(T).txt', delimiter=';')
T_table_c = data_c[:, 0]  # teploty (K) pre c
c_table = data_c[:, 1]  # špecifické tepelné kapacity (J/kg.K)

data_h0 = np.loadtxt('H_Flux_h0.txt', delimiter=';')
T_table_h0 = data_h0[:, 0]  # teploty (K) pre krivku h0
h0_table = data_h0[:, 1]  # koeficienty odvodu tepla (W/m^2.K)

data_h1 = np.loadtxt('H_Flux_h1.txt', delimiter=';')
T_table_h1 = data_h1[:, 0]  # teploty (K) pre krivku h1
h1_table = data_h1[:, 1]  # koeficienty odvodu tepla (W/m^2.K)

data_q0 = np.loadtxt('HeatFlux_q0.txt', delimiter='\t')
T_table_q0 = data_q0[:, 0]  # teploty (K) pre krivku q0
q0_table = data_q0[:, 1]  # hustota tepelného toku (W/m^2)

data_I = np.loadtxt(file_path_I, delimiter='\t')
t_table_I = data_I[:, 0]  # časy (s) pre el. pulz I
I_table = data_I[:, 1]  # prúdy el. pulzu (A)

# Nastavenie t_max ako posledná hodnota z t_table_I
t_max = t_table_I[-1]

# Interpolácie
R_interp = interp1d(T_table_R, R_table, kind='linear', fill_value='extrapolate')
c_interp = interp1d(T_table_c, c_table, kind='linear', fill_value='extrapolate')
h0_interp = interp1d(T_table_h0, h0_table, kind='linear', fill_value='extrapolate')
h1_interp = interp1d(T_table_h1, h1_table, kind='linear', fill_value='extrapolate')
q0_interp = interp1d(T_table_q0, q0_table, kind='linear', fill_value='extrapolate')
I_interp = interp1d(t_table_I, I_table, kind='linear', fill_value='extrapolate')

# Inicializácia zoznamov s počiatočnými hodnotami
T_values = [T_initial]  # Počiatočná teplota vodiča
R_values = [R_interp(T_initial)]  # Odpor pri počiatočnej teplote
c_values = [c_interp(T_initial)]  # Špecifická tepelná kapacita pri počiatočnej teplote
h_values = [h0_interp(T_initial)]  # Koeficient odvodu tepla pre počiatočnú teplotu
q_values = [q0_interp(T_initial)]  # Hustota tepelného toku pre počiatočnú teplotu
I_values = [I_interp(0)]  # Prúd v čase t = 0

# Inicializácia logickej premennej
q_gen_exceeded = False

# Definovanie diferenciálnej rovnice
def dT_dt(t, T, A, m, T_env, R_interp, c_interp, I_interp, q0_interp, h0_interp, h1_interp):
    R = R_interp(T)
    c = c_interp(T)
    I = I_interp(t)
    q0 = q0_interp(T)

    P_in = I**2 * R
    T_gen = P_in / (m * c)
    q_gen = P_in / A
    
    q_gen_exceeded = q_gen >= q0
    
    h = h1_interp(T) if q_gen_exceeded else h0_interp(T)
    h = min(h, 120000)

    T_out = (h * A * (T - T_env)) / (m * c)
    
    return T_gen - T_out

times = np.linspace(0, t_max, int(t_max / dt) + 1)  # časové body

# Numerické riešenie diferenciálnej rovnice pomocou solve_ivp
args = (A, m, T_env, R_interp, c_interp, I_interp, q0_interp, h0_interp, h1_interp)
solution = solve_ivp(dT_dt, [0, t_max], [T_initial], method='LSODA', args=args, t_eval=times)
T_values = solution.y[0]

# solve_ivp poskytuje rôzne metódy riešenia, ktoré môžu byť lepšie pre rôzne typy rovníc.
# Naríklad "RK45", "RK23", "DOP853", "Radau", "BDF", alebo "LSODA"

# Vykreslenie výsledkov
plt.plot(times, T_values, label='Teplota')
plt.xlabel('Čas (s)')
plt.ylabel('Teplota (K)')
plt.title('Priebeh teploty Cu vodiča v závislosti od času')
plt.legend()
plt.grid()
plt.show()
