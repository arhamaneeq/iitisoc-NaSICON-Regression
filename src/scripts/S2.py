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

def normaliseCompositions(comp1, comp2, m2):
    for el in comp1:
        if el.symbol not in M and el in comp2:
            k = comp1[el] / comp2[el]
            break
    else:
        return comp2, m2 

    scaledComp2 = Composition({el: amt * k for el, amt in comp2.items()})
    scaled_m2 = m2 * k

    return scaledComp2, scaled_m2


groups = defaultdict(list)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping Materials"):
    mID = row["material_id"]
    formula = row["composition"]
    energy_per_atom = row["energy_per_atom"]

    try:
        framework_formula, m_count = getFramework(formula)
        groups[framework_formula].append((mID, formula, m_count, energy_per_atom))
    except Exception as e:
        #print(f"Skipping {row['material_id']} ({formula}) due to parsing error: {e}")
        continue

pairs = []
for framework, entries in tqdm(groups.items(), desc="Pairing Frameworks"):
    entries = sorted(entries, key=lambda x: x[2])  

    for (mid1, f1, m1, e1), (mid2, f2, m2, e2) in combinations(entries, 2):
        comp1 = Composition(f1)
        comp2 = Composition(f2)

        comp2_scaled, m2_scaled = normaliseCompositions(comp1, comp2, m2)
        f2_scaled_str = comp2_scaled.formula

        _, dM = getMetals(f1, f2_scaled_str, M)

        if m1 != m2_scaled: # and e1 == e2:
            pairs.append({
                "framework": framework,
                "charged_id": mid2,
                "charged_formula": f2_scaled_str,
                "charged_m": m2_scaled,

                "discharged_id": mid1,
                "discharged_formula": f1,
                "discharged_m": m1,

                "active_metals": dM,
                "m_count_diff": m2_scaled - m1,

                "energy_per_atom": e1,
            })

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv("data/DS2.csv", index=False)
