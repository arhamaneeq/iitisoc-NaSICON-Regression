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

FARADAY = 96485.33212
MU = {
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

df = pd.read_csv("data/DS2.csv")
delEs = []
delVs = []

for row in tqdm(df.itertuples(index=False), desc="Calculating Energies", total=len(df)):
    M = row.active_metals
    n = row.m_count_diff

    if M not in MU:
        delEs.append(None)
        delVs.append(None)
        continue

    E_charged = row.charged_energy * Composition(row.charged_formula).num_atoms
    E_discharged = row.discharged_energy * Composition(row.discharged_formula).num_atoms

    delE = E_charged - E_discharged + n * MU[M]
    delV = - (delE * 1.60218e-19 ) / (n * 1 * FARADAY) # 1.60218e-19 J = 1eV

    delEs.append(delE)
    delVs.append(delV)

df["delta_E"] = delEs
df["delta_V"] = delVs

df.to_csv("data/DS3.csv")