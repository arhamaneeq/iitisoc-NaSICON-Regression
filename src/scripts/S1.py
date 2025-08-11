from mp_api.client import MPRester
import pandas as pd
from tqdm import tqdm
import json

with open("keys.json") as f:
    secrets = json.load(f)
API_KEY = secrets["materials_project"]["api_key"]


# Storage
materials = []

FIELDS = [
    "material_id", # "symmetry", "nsites", #"structure",
    "composition_reduced", "energy_above_hull", "energy_per_atom", # "volume", # "elements", 
    "is_stable", #"dimensionality", #"structure.dimensionality"
]

materials = []

with MPRester(API_KEY) as mpr:
    for doc in mpr.materials.summary.search(
        deprecated=False,
        fields=FIELDS,
        chunk_size=1000,
        num_chunks=None
    ):
        try:
            materials.append(doc)
        except Exception as e:
            print(f"Skipping document due to error: {e}")

df = pd.DataFrame([{
    "material_id": m.material_id,
    #"spacegroup_symbol": m.symmetry.symbol if m.symmetry else None,
    #"spacegroup_number": m.symmetry.number if m.symmetry else None,
    #"crystal_system": m.symmetry.crystal_system if m.symmetry else None,
    #"nsites": m.nsites,
    #"elements": m.elements,
    "composition": m.composition_reduced,
    "energy_per_atom": m.energy_per_atom,
    #"volume": m.volume,
    #"is_stable": m.is_stable,
    #"dimensionality": m.dimensionality,
} for m in materials if (m.is_stable or m.energy_above_hull < 0.05)]).reset_index(drop=True)


# df.to_pickle("data/NaSICON_Dataset_14.pkl")
df.to_csv   ("data/DS1.csv")

print(f"Downloaded {len(df)} crystalline solids")