# -*- coding: utf-8 -*-
"""
Program na výpočet ohrevu vodiča prúdom I ponoreného do LN2
Vylepšená verzia: robustné načítanie dát, bezpečné interpolácie, ošetrenie GUI a chýb.
Created on Fri Dec 27 15:40:46 2024
@author: Lubos (upravené)
"""
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button, font, Toplevel, messagebox

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ParameterDialog:
    def __init__(self, master, params=None):
        # master should be a single Tk() root; dialog is a Toplevel
        self.master = master
        self.dialog = Toplevel(master)
        self.dialog.title("Nastavenie parametrov")
        self.entries = {}

        # Load parameters from file if available
        if params is None:
            params = self.load_params()

        self.params = params

        # Build form
        row = 0
        for label, default in self.params.items():
            tk_label = Label(self.dialog, text=label)
            tk_label.grid(row=row, column=0, sticky="w", padx=10, pady=5)

            entry = Entry(self.dialog)
            entry.insert(0, str(default))
            entry.grid(row=row, column=1, padx=10, pady=5)
            self.entries[label] = entry

            row += 1

        bold_font = font.Font(weight="bold")
        confirm_btn = Button(self.dialog, text="OK", font=bold_font,
                             command=self.on_confirm, width=8, height=1)
        confirm_btn.grid(row=row, column=0, columnspan=2, pady=10)

        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.cancelled = False

    def on_confirm(self):
        params = {}
        try:
            for label, entry in self.entries.items():
                # All parameters expected as floats here
                params[label] = float(entry.get())
        except ValueError:
            messagebox.showerror("Chyba", "Niekde je neplatná číselná hodnota. Skontrolujte vstupy.")
            return
        self.params = params
        self.save_params()
        self.dialog.destroy()

    def on_cancel(self):
        self.cancelled = True
        self.dialog.destroy()

    def get_parameters(self):
        self.master.wait_window(self.dialog)
        if self.cancelled:
            return None
        return self.params

    def save_params(self):
        try:
            with open("params.json", "w", encoding="utf-8") as file:
                json.dump(self.params, file, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning("Nepodarilo sa uložiť params.json: %s", e)

    def load_params(self):
        if os.path.exists("params.json"):
            try:
                with open("params.json", "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception as e:
                logging.warning("Chyba pri načítaní params.json, použijem default: %s", e)
        # Default parameters (bez diakritiky v kľúčoch, ak chcete jednoduchší JSON použite ascii kluce)
        return {
            "Hmotnost vodica (m) [kg]": 0.0001995,
            "Teplota okolia (T_env) [K]": 77.0,
            "Pociatocna teplota vodiča (T_initial) [K]": 77.0,
            "Plocha vodica (A) [m^2]": 8.2693E-5,
            "Casovy krok (dt) [s]": 0.001,
            "Maximalny h (max_h) [W/m^2.K]": 120000.0,
        }


def try_loadtxt(path, delimiters=(';', '\t', None)):
    """Skúsi načítať textový súbor s viacerými delimitermi, vráti numpy array alebo vyhodí ValueError/FileNotFoundError."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Súbor neexistuje: {path}")
    last_exc = None
    for d in delimiters:
        try:
            if d is None:
                data = np.loadtxt(path)
            else:
                data = np.loadtxt(path, delimiter=d)
            # Ensure at least two columns
            if data.ndim == 1 or data.shape[1] < 2:
                raise ValueError("Očakávam aspoň 2 stĺpce v dátach.")
            return data
        except Exception as e:
            last_exc = e
    raise ValueError(f"Nepodarilo sa načítať súbor {path}: {last_exc}")


def safe_interp_factory(x, y, name="", saturate=True, default_out=None):
    """
    Vytvorí interp function, ktorá použije saturáciu na okrajé hodnoty (ak saturate=True).
    Ak default_out je zadané a x je jednoprvkové, vráti konštantnú funkciu.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1:
        raise ValueError("x musí byť 1D")
    if x.size == 1:
        # Degenerate case: return constant
        def const_fn(xq):
            return float(y[0])
        return const_fn

    if saturate:
        interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
    else:
        interp = interp1d(x, y, kind='linear', fill_value='extrapolate', bounds_error=False)

    def f(q):
        q = np.asarray(q)
        return interp(q)
    return f


def main():
    # Single root for all dialogs
    root = tk.Tk()
    root.withdraw()

    # Parameter dialog
    param_dialog = ParameterDialog(root)
    params = param_dialog.get_parameters()
    if params is None:
        logging.info("Používateľ zrušil dialóg s parametrami. Končím.")
        return

    m = params["Hmotnost vodica (m) [kg]"]
    T_env = params["Teplota okolia (T_env) [K]"]
    T_initial = params["Pociatocna teplota vodiča (T_initial) [K]"]
    A = params["Plocha vodica (A) [m^2]"]
    dt = params["Casovy krok (dt) [s]"]
    max_h = params.get("Maximalny h (max_h) [W/m^2.K]", 120000.0)

    # Select I(t) file
    file_path_I = filedialog.askopenfilename(title="Vyber súbor I(t)", filetypes=[("Textové súbory", "*.txt;*.dat;*.tsv"), ("All files", "*.*")])
    if not file_path_I:
        logging.info("Nebola vybraná žiadna cesta k súboru I(t). Končím.")
        return

    # Load data tables with robust error handling
    try:
        data_R = try_loadtxt('Copper_R(T).txt', delimiters=(';', '\t', None))
        T_table_R = data_R[:, 0]
        R_table = data_R[:, 1]

        data_c = try_loadtxt('Coeff_Cp(T).txt', delimiters=(';', '\t', None))
        T_table_c = data_c[:, 0]
        c_table = data_c[:, 1]

        data_h0 = try_loadtxt('H_Flux_h0.txt', delimiters=(';', '\t', None))
        T_table_h0 = data_h0[:, 0]
        h0_table = data_h0[:, 1]

        data_h1 = try_loadtxt('H_Flux_h1.txt', delimiters=(';', '\t', None))
        T_table_h1 = data_h1[:, 0]
        h1_table = data_h1[:, 1]

        data_q0 = try_loadtxt('HeatFlux_q0.txt', delimiters=('\t', ';', None))
        T_table_q0 = data_q0[:, 0]
        q0_table = data_q0[:, 1]

        data_I = try_loadtxt(file_path_I, delimiters=('\t', ';', None))
        t_table_I = data_I[:, 0]
        I_table = data_I[:, 1]
    except Exception as e:
        logging.error("Chyba pri načítaní dátových súborov: %s", e)
        return

    # Create safe interpolators (saturate on edges to avoid crazy extrapolation)
    R_interp = safe_interp_factory(T_table_R, R_table, name="R", saturate=True)
    c_interp = safe_interp_factory(T_table_c, c_table, name="c", saturate=True)
    h0_interp = safe_interp_factory(T_table_h0, h0_table, name="h0", saturate=True)
    h1_interp = safe_interp_factory(T_table_h1, h1_table, name="h1", saturate=True)
    q0_interp = safe_interp_factory(T_table_q0, q0_table, name="q0", saturate=True)

    # For current I(t) prefer zero outside given time range
    I_interp = interp1d(t_table_I, I_table, kind='linear', bounds_error=False, fill_value=0.0)

    # Time settings
    t_max = float(t_table_I[-1])
    times = np.linspace(0, t_max, int(max(2, np.ceil(t_max / dt))) + 1)

    # Differential equation
    def dT_dt(t, T):
        # T is array-like from solve_ivp
        T = float(T[0]) if np.ndim(T) else float(T)
        R = float(R_interp(T))
        c = float(c_interp(T))
        I = float(I_interp(t))
        q0 = float(q0_interp(T))

        # Safety clamps
        c = max(c, 1e-8)   # zabraňuje deleniu nulou
        R = max(R, 1e-12)

        P_in = I ** 2 * R           # W
        T_gen = P_in / (m * c)      # K/s (prírastok z generovaného výkonu)
        q_gen = P_in / A            # W/m^2

        # Koeficient odvodu tepla závisí na tom, či q_gen >= q0 (tak, ako v pôvodnom kóde)
        q_gen_exceeded = q_gen >= q0
        h = float(h1_interp(T)) if q_gen_exceeded else float(h0_interp(T))
        h = min(h, max_h)

        T_out = (h * A * (T - T_env)) / (m * c)  # strata tepla (K/s)
        dT = T_gen - T_out
        return dT

    # solve_ivp expects function(t, y)
    try:
        solution = solve_ivp(lambda t, y: [dT_dt(t, y)], [0, t_max], [T_initial],
                             method='LSODA', t_eval=times, dense_output=True, atol=1e-8, rtol=1e-6)
        if not solution.success:
            logging.warning("solve_ivp neúspešné: %s", solution.message)
    except Exception as e:
        logging.error("Chyba pri riešení ODE: %s", e)
        return

    T_values = solution.y[0]
    # Prepare I(t) sampled on the same times for plotting
    I_values = I_interp(times)

    # Plot results: T(t) a I(t) na druhej osi
    fig, ax1 = plt.subplots()
    ax1.plot(times, T_values, color='tab:red', label='Teplota (K)')
    ax1.set_xlabel('Čas (s)')
    ax1.set_ylabel('Teplota (K)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(times, I_values, color='tab:blue', linestyle='--', label='Prúd (A)')
    ax2.set_ylabel('Prúd (A)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Priebeh teploty a prúdu v čase')
    fig.tight_layout()

    # Save figure next to I file
    out_fig = os.path.splitext(os.path.basename(file_path_I))[0] + "_result.png"
    try:
        fig.savefig(out_fig, dpi=150)
        logging.info("Graf uložený ako %s", out_fig)
    except Exception as e:
        logging.warning("Nepodarilo sa uložiť graf: %s", e)

    plt.show()


if __name__ == "__main__":
    main()
