from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm
import json
from pymatgen.core import Composition
from itertools import combinations
from collections import defaultdict
from pymatgen.core.periodic_table import Element

with open("keys.json") as f:
    secrets = json.load(f)
API_KEY = secrets["materials_project"]["api_key"]

FARADAY = 96485.33212       # Coulumbs (1 Na * QE)
QE      = 1.60218e-19       # Coulombs
MU = {                      # eV/atom (chemical potentials)
    "Li": -1.90,
    "Na": -1.14,
    "K": -1.00,
    "Rb": -0.85,
    "Cs": -0.80,
    "Mg": -1.50,
    "Ca": -1.60,
    "Al": -3.74,
    "Zn": -1.30,
    "Y": -3.40
}

MM = {                      # g / mol
    "Li": 6.94,
    "Na": 22.99,
    "K": 39.098,
    "Rb": 85.468,
    "Cs": 132.91,
    "Fr": 223,
    "Mg": 24.305,
    "Ca": 40.078,
    "Al": 26.982,
    "Zn": 65.38,
    "Y": 88.906
}

def valence(symbol):
    el = Element(symbol)
    if el.oxidation_states:
        return el.oxidation_states[0]
    else:
        raise ValueError(f"No oxidation state for element {symbol}")

df = pd.read_csv("data/DS2.csv")

delVs = []
speCs = []
speEs = []

for row in tqdm(df.itertuples(index=False), desc="Calculating Energies", total=len(df)):
    metals = row.active_metals.split("|") if row.active_metals else []
    MM_host = Composition(row.discharged_formula).weight

    if not metals:
        delVs.append(None)
        speCs.append(None)
        speEs.append(None)
        continue

    per_metal_diff = row.m_count_diff / len(metals) if len(metals) > 0 else 0

    total_n = 0
    total_mu_term = 0
    for m in metals:
        n_metal = per_metal_diff * valence(m)
        total_n += n_metal
        total_mu_term += n_metal * MU[m]

    if total_n == 0:
        delVs.append(None)
        speCs.append(None)
        speEs.append(None)
        continue

    E_charged = row.charged_energy_total_scaled
    E_discharged = row.discharged_energy_total

    delV = - (E_charged - E_discharged - total_mu_term) / total_n
    speC = (total_n * FARADAY * 1000) / (3600 * MM_host)  # mAh / g
    speE = speC * delV                                    # mWh / g

    if abs(delV) > 10:
        delVs.append(None)
        speCs.append(None)
        speEs.append(None)
        continue 

    delVs.append(delV)
    speCs.append(speC)
    speEs.append(speE)


df["specific_capacity"] = speCs
df["delta_V"] = delVs
df["specific_energy"] = speEs

df.to_csv("data/DS3.csv")
