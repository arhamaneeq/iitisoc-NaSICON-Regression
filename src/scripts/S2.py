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

df = pd.read_csv("data/DS1.csv")

M = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Al", "Zn", "Y"}

def getFramework(formula):
    C = Composition(formula)
    
    framework = {element: amount for element, amount in C.items() if element.symbol not in M}
    countM = sum(amount for element, amount in C.items() if element.symbol in M)

    frameworkC = Composition(framework)
    frameworkF = frameworkC.reduced_formula

    return frameworkF, countM

def getMetals(charged_formula, discharged_formula, M):
    charged = Composition(charged_formula)
    discharged = Composition(discharged_formula)
    metals = [el.symbol for el in charged.keys() | discharged.keys() if el.symbol in M]
    return len(metals), "|".join(sorted(set(metals)))

groups = defaultdict(list)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping Materials"):
    mID = row["material_id"]
    formula = row["composition"]
    energy = row["energy"]

    try:
        framework_formula, m_count = getFramework(formula)
        groups[framework_formula].append((mID, formula, m_count, energy))
    except Exception as e:
        #print(f"Skipping {row['material_id']} ({formula}) due to parsing error: {e}")
        continue

pairs = []

for framework, entries in tqdm(groups.items(), desc="Pairing Frameworks"):
    entries = sorted(entries, key = lambda x: x[2])

    for (mid1, f1, m1, e1), (mid2, f2, m2, e2) in combinations(entries, 2):
        _, dM = getMetals(f1, f2, M)

        if (m1 != m2):
            pairs.append({
                "framework": framework,
                "charged_id": mid2,
                "charged_formula": f2,
                "charged_energy": e2,
                "charged_m": m2,

                "discharged_id": mid1,
                "discharged_formula": f1,
                "discharged_energy": e1,
                "discharged_m": m1,

                "active_metals": dM,
                "m_count_diff": m2 - m1
            })

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv("data/DS2.csv")