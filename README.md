# AI-Driven Discovery of High-Performance NaSICON Cathode Materials for Sodium-Ion Batteries

The development of high-energy density cathode materials remains a considerable challenge for the advent of cost-effective sodium-ion batteries in their bid to replace traditional lithium-ion batteries. Among various candidates, a class of promising contenders are sodium superionic conductors (NaSICON)-type materials, which may occur with various dopants, substitutions, or stoichiometries, each with distinct properties. Given the vastness of the candidate space, experimental methods for optimization are computationally expensive and practically intractable. We propose a modular machine learning-based framework to classify and regress material properties from known features, and to systematically generate a searchable database for material lookup and comparison.

## Overview
### Team
- `Arham Aneeq`, BTech MEMS, IIT Indore 2028
- `Shriram Naik`, BTech MEMS, IIT Indore 2028
- `Harshavardhan Pawar`, BTech MEMS, IIT Indore 2028
- `Shrawan Govindwar`, BTech MEMS, IIT Indore 2028

**Mentor**
- `Mohak Dadhich`, BTech CHE, IIT Indore 2027

### Project
- **Domain:** Metacryst
- **PS Name:** AI-Driven Discovery of High-Performance NaSICON Cathode Materials for Sodium-Ion Batteries
- **Team Number:** MM-001 

## Implementation

### Repository Structure
```bash
.
├── data/                  # Processed datasets at each pipeline stage
│   ├── DS1.csv            # Stable material dataset from Materials Project
│   ├── DS2.csv            # Grouped and paired charged/discharged materials
│   ├── DS3.csv            # Calculated electrochemical properties
│   └── DS4.csv            # Featurized dataset for ML training
├── models/                # Saved ML models and training logs
│   ├── DNN1.pth           # Trained voltage prediction model weights
│   └── training_history.csv # Training metrics per epoch
├── keys.json              # API keys (not tracked in VCS)
├── src/
│   ├── scripts/           # Pipeline scripts
│   │   ├── S1_download.py     # Data download & initial filtering
│   │   ├── S2_group_pair.py   # Framework grouping and pairing
│   │   ├── S3_calc_energy.py  # Voltage and capacity calculations
│   │   ├── S4_featurize.py    # Feature engineering using matminer
│   │   └── S5_train_model.py  # Deep neural network training
│   └── notebooks/          # Exploratory data analysis and model analysis
│       ├── NB1_EDA.ipynb   # Initial dataset exploration
│       ├── NB2_EDA.ipynb   # Further exploratory analysis and visualization
│       └── NB3_MA.ipynb    # Model analysis and performance evaluation
└── README.md               # This documentation
```


### Pipeline
#### Data Acquisition & Filtering
- Uses `mp-api` to query the Materials Project database.
- Downloads stable or near-stable crystalline solids with relevant properties such as `material_id`, `composition_reduced`, `energy_per_atom`, and `energy_above_hull`.
- Filters for stability (is_stable=True) or small energy above hull (< 0.05 eV/atom).
- Saves raw dataset to `data/DS1.csv`.

#### Framework Grouping and Charged/Discharged Pairing

- Reads DS1.csv.
- Identifies the framework of each material by removing alkali and alkaline earth metals (set M).
- Groups materials by their reduced framework formula.
- Creates pairs of materials differing by metal content to simulate charged/discharged battery states.
- Normalizes compositions to account for scaling.
- Computes differences in metal count and total energies.
- Outputs paired data to data/DS2.csv.

#### Electrochemical Property Calculation
- Reads paired dataset DS2.csv.
- Uses tabulated chemical potentials (μ) and molar masses for metals (Li, Na, K, etc.).

Calculates:

- Average voltage (ΔV) using total energies and chemical potentials.
- Specific capacity (mAh/g).
- Specific energy (mWh/g).
- Filters physically unreasonable voltage values (>10 V).
- Saves enhanced dataset to data/DS3.csv.

#### Feature Engineering
- Loads DS3.csv.
- Converts chemical formulas into pymatgen Composition objects.
- Applies several matminer composition-based featurizers:
- Magpie elemental properties
- Stoichiometry features
- Valence orbital occupations
- Outputs the fully featurized dataset data/DS4.csv ready for ML.

#### Deep Neural Network Training
- Loads featurized data from DS4.csv.
- Cleans dataset by removing infinite and missing values.
- Splits data into train/test (80/20) sets.
- Normalizes features using StandardScaler.
- Defines a PyTorch feed-forward neural network to regress voltage (delta_V).
- Uses Adam optimizer, MSE loss, and early stopping (patience=20 epochs).
- Trains for up to 100 epochs with ReLU activations.
- Saves best model weights (models/DNN1.pth) and training metrics log (training_history.csv).

## Setup
### GitHub
Clone the repository using
```bash
git clone https://github.com/arhamaneeq/iitisoc-NaSICON-Regression.git
cd iitisoc-NaSICON-Regression
```
### Environment
In your root directory create a virtual environment using
```bash
python -m venv .venv
.venv/Scripts/Activate
pip install -r requirements.txt -q
```
You can update the `requirements.txt` file by writing
```bash
pip freeze > requirements.txt
```

> [!IMPORTANT]
> Do not modify `requirements.txt` directly in any commit to the main branch. If you believe dependencies must be installed, create a new branch, install and test dependencies, update `requirements.txt`, and then open a Pull Request.

### Secrets
Save your secrets in a `keys.json` file in the root folder. Current structure of `keys.json` file is:
```json
{
    "materials_project": {
        "api_key": "NEVERGONNAGIVEYOUUP"
    }
}
```

> [!WARNING]
> Never expose your API keys.

> [!IMPORTANT]
> Update this readme if new keys are added to `keys.json`.
