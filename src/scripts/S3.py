from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm
import json
from pymatgen.core import Composition
from itertools import combinations
from collections import defaultdict

with open("keys.json") as f:
    secrets = json.load(f)
API_KEY = secrets["materials_project"]["api_key"]

FARADAY = 96485.33212       # Coulumbs (1 Na * QE)
QE      = 1.60218e-19       # Coulombs
MU = {                      # 
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

df = pd.read_csv("data/DS2.csv")

delVs = []
speCs = []
speEs = []

for row in tqdm(df.itertuples(index=False), desc="Calculating Energies", total=len(df)):
    M = row.active_metals
    n = row.m_count_diff

    MM_host = Composition(row.discharged_formula).weight

    if M not in MU:
        delVs.append(None)
        speCs.append(None)
        speEs.append(None)
        continue

    E_charged = row.charged_energy * Composition(row.charged_formula).num_atoms
    E_discharged = row.discharged_energy * Composition(row.discharged_formula).num_atoms

    delV = - (E_charged - E_discharged + n * MU[M]) / (n)   # Volt
    speC = (n * FARADAY * 1000) / (3600 * MM_host)            # mAh / g
    speE = speC * delV                                      # mWh / g

    delVs.append(delV)
    speCs.append(speC)
    speEs.append(speE)

df["specific_capacity"] = speCs
df["delta_V"] = delVs
df["specific_energy"] = speEs

df.to_csv("data/DS3.csv")