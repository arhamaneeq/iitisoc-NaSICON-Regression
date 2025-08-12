import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    OxidationStates
)
from tqdm import tqdm

tqdm.pandas()

def safe_add_oxidation_states(formula):
    try:
        comp = Composition(formula)
        return comp.add_charges_from_oxi_state_guesses()
    except Exception as e:
        return None
    
def safe_make_composition(formula):
    try:
        return Composition(formula)
    except Exception:
        return None

def featurize_dataset(input_csv, output_csv):
    # Load original dataset
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} entries from {input_csv}")

    df["composition"] = df["discharged_formula"].progress_apply(safe_make_composition)
    # Guess oxidation states
    # df["composition"] = df["discharged_formula"].progress_apply(safe_add_oxidation_states)
    #print(f"{len(df)} entries retained after oxidation state guessing")

    df = df[df["composition"].notnull()].reset_index(drop=True)
    print(f"{len(df)} entries retained after making compositions")

    # Apply featurizers
    featurizers = [
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
        ValenceOrbital(),
        # OxidationStates()
    ]

    for f in featurizers:
        print(f"Applying: {f.__class__.__name__}")

        features = df["composition"].progress_apply(
            lambda comp: f.featurize(comp)
        )

        feature_labels = f.feature_labels()
        feature_df = pd.DataFrame(features.tolist(), columns=feature_labels)
        df = pd.concat([df, feature_df], axis=1)

    # Save result
    df.to_csv(output_csv, index=False)
    print(f"Saved featurized dataset with {df.shape[1]} columns to {output_csv}")

if __name__ == "__main__":
    featurize_dataset(
        input_csv="data/DS3.csv",
        output_csv="data/DS4.csv"
    )