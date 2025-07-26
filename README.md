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
