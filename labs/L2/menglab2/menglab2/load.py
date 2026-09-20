from __future__ import annotations

import io
import os
import tarfile
import tempfile
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors


def load_full_dataset(save_csv: bool = False) -> pd.DataFrame:
    """
    Download, curate, and return a molecular boiling-point dataset with RDKit descriptors.

    This function assembles a combined dataset from two sources:

    1) **bpdata** from the CRAN package `rcdk` (downloaded from the official tarball).
       - The file `bpdata.rda` is extracted and read via `pyreadr`.
       - Columns are normalized to a common schema and Celsius is derived from Kelvin.

    2) **Curated alkanes and branched isomers** with canonical SMILES and typical
       boiling points at 1 atm.

    For each molecule, basic RDKit descriptors are computed from the SMILES string:
      - MW (molecular weight, g/mol)
      - LogP (octanol/water partition coefficient, XlogP3 approximation)
      - TPSA (topological polar surface area)
      - RotatableBonds (number of rotatable bonds)
      - HBD (hydrogen bond donors)
      - HBA (hydrogen bond acceptors)
      - OxygenCount (count of oxygen atoms)

    Parameters
    ----------
    save_csv : bool, default=False
        If True, write CSVs with intermediate and final datasets into the
        current working directory. Files written are:
          - `bpdata_full_with_descriptors.csv`
          - `alkanes_curated_with_descriptors.csv`
          - `combined_alkane_dataset.csv`

    Returns
    -------
    pandas.DataFrame
        A combined dataset with the following columns (where available):
        ['name', 'smiles', 'bp_k', 'bp_c',
         'MW', 'LogP', 'TPSA', 'RotatableBonds', 'HBD', 'HBA', 'OxygenCount',
         'branching_index', 'BranchingIndex_Kappa1'].

    Raises
    ------
    requests.HTTPError
        If the CRAN tarball cannot be downloaded successfully.
    FileNotFoundError
        If `bpdata` is not found inside the downloaded tarball.
    """

    # ---------- Helper: compute RDKit descriptors ----------
    def _compute_descriptors(smiles: str) -> List[Optional[float]]:
        """Return [MW, LogP, TPSA, RotatableBonds, HBD, HBA, OxygenCount] or Nones if SMILES is invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * 7

        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8),
        ]

    def branching_index_from_smiles(smiles: str) -> int:
      """
      Calculate a simple branching index for a molecule from its SMILES string.

      Definition:
        For each heavy atom, if degree > 2, add (degree - 2) to the index.

      Returns 0 if SMILES parsing fails.
      """
      mol = Chem.MolFromSmiles(smiles)
      if mol is None:
          return 0  # could also return None

      index = 0
      for atom in mol.GetAtoms():
          if atom.GetAtomicNum() == 1:  # skip hydrogens
              continue
          degree = atom.GetDegree()
          if degree > 2:
              index += (degree - 2)
      return index


    # ============================================================
    # 1) Load bpdata from CRAN tarball (dataset shipped in `rcdk`)
    # ============================================================
    #url = "https://cran.r-project.org/src/contrib/rcdk_3.8.1.tar.gz"
    url = "https://github.com/andrewlferguson/MENG15100/blob/main/labs/L2/rcdk_3.8.2.tar.gz"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
        member = None
        for m in tf.getmembers():
            nm = m.name.lower()
            if nm.endswith("/data/bpdata.rda") or nm.endswith("/data/bpdata.rdata"):
                member = m
                break
        if member is None:
            raise FileNotFoundError("bpdata not found in rcdk tarball")

        rdata_bytes = tf.extractfile(member).read()  # type: ignore[union-attr]

    # Read the .rda into a temporary file with pyreadr
    import pyreadr

    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as tmp:
        tmp.write(rdata_bytes)
        tmp_path = tmp.name

    try:
        res = pyreadr.read_r(tmp_path)
        df_bpdata_raw = res["bpdata"] if "bpdata" in res else list(res.values())[0]

        # Normalize columns and add Celsius
        df_bpdata_raw = (
            df_bpdata_raw.reset_index().rename(
                columns={"rownames": "name", "SMILES": "smiles", "BP": "bp_k"}
            )
        )
        df_bpdata_raw["bp_c"] = df_bpdata_raw["bp_k"] - 273.15
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Compute descriptors for bpdata
    df_bpdata_desc = df_bpdata_raw.copy()
    desc_cols = ["MW", "LogP", "TPSA", "RotatableBonds", "HBD", "HBA", "OxygenCount"]
    df_bpdata_desc[desc_cols] = df_bpdata_desc["smiles"].apply(
        lambda s: pd.Series(_compute_descriptors(s))
    )
    df_bpdata_desc["branching_index"] = None
    df_bpdata_desc["BranchingIndex_Kappa1"] = None

    # ============================================================
    # 2) Curated alkanes (+ branched isomers) with canonical SMILES
    # ============================================================
    n_alkanes = [
        ("methane", 1, -161.5),
        ("ethane", 2, -88.6),
        ("propane", 3, -42.1),
        ("butane", 4, -0.5),
        ("pentane", 5, 36.1),
        ("hexane", 6, 68.7),
        ("heptane", 7, 98.4),
        ("octane", 8, 125.7),
        ("nonane", 9, 150.8),
        ("decane", 10, 174.1),
    ]
    df_n = pd.DataFrame(n_alkanes, columns=["name", "nC", "bp_c"])
    df_n["bp_k"] = df_n["bp_c"] + 273.15
    df_n["branching_index"] = 0
    df_n["smiles"] = df_n["nC"].apply(lambda n: "C" * int(n))

    branched = pd.DataFrame(
        [
            ("isobutane", 4, -11.7, 1, "CC(C)C"),
            ("isopentane", 5, 27.8, 1, "CC(C)CC"),
            ("neopentane", 5, 9.5, 2, "CC(C)(C)C"),
            ("isooctane", 8, 99.2, 3, "CC(C)CC(C)(C)C"),
            ("isohexane", 6, 60.3, 1, "CC(C)CCC"),
            ("neohexane", 6, 49.7, 2, "CC(C)(C)CC"),
            ("isoheptane", 7, 90.6, 1, "CC(C)CCCC"),
            ("neoheptane", 7, 79.2, 2, "CC(C)(C)CCC"),
        ],
        columns=["name", "nC", "bp_c", "branching_index", "smiles"],
    )
    branched["bp_k"] = branched["bp_c"] + 273.15

    df_alkanes_curated = pd.concat([df_n, branched], ignore_index=True)
    df_alkanes_curated[desc_cols] = df_alkanes_curated["smiles"].apply(
        lambda s: pd.Series(_compute_descriptors(s))
    )
    df_alkanes_curated["BranchingIndex_Kappa1"] = None

    drop_cols = [c for c in ["nC", "nH"] if c in df_alkanes_curated.columns]
    df_alkanes_desc = df_alkanes_curated.drop(columns=drop_cols)
    ordered_cols = [
        "name",
        "smiles",
        "bp_k",
        "bp_c",
        "MW",
        "LogP",
        "TPSA",
        "RotatableBonds",
        "HBD",
        "HBA",
        "OxygenCount",
        "branching_index",
        "BranchingIndex_Kappa1",
    ]
    df_alkanes_desc = df_alkanes_desc[ordered_cols]

    # ============================================================
    # 3) Merge bpdata + curated alkanes/isomers
    # ============================================================
    keep_cols = ordered_cols

    for col in keep_cols:
        if col not in df_bpdata_desc.columns:
            df_bpdata_desc[col] = None
    df_bpdata_desc = df_bpdata_desc[keep_cols]

    df_alkanes_combined = pd.concat(
        [df_bpdata_desc[keep_cols], df_alkanes_desc[keep_cols]], ignore_index=True
    )

    # add branching index 
    df_alkanes_combined["branching_index"] = df_alkanes_combined["smiles"].apply(branching_index_from_smiles)
    # add log MW as own feature column
    df_alkanes_combined['logMW'] = np.log(df_alkanes_combined['MW'])

    if save_csv:
        df_bpdata_desc.to_csv("bpdata_full_with_descriptors.csv", index=False)
        df_alkanes_desc.to_csv("alkanes_curated_with_descriptors.csv", index=False)
        df_alkanes_combined.to_csv("combined_alkane_dataset.csv", index=False)

    return df_alkanes_combined

def load_full_train_test_split():
  from sklearn.model_selection import train_test_split
  df = load_full_dataset()

  # Select features (molecular properties) and target (boiling point)
  # We will use numerical columns as features, excluding the target itself
  feature_cols = ['MW', 'logMW', 'branching_index', 'LogP', 'TPSA', 'RotatableBonds', 'HBD', 'HBA', 'OxygenCount']

  # Combine features and target for dropping rows with missing values
  combined_data = df[feature_cols + ['bp_k']].dropna()

  X = combined_data[feature_cols] # Features
  y = combined_data['bp_k'] # Target

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test

def load_dataset():
    """
    Load a simplified molecular dataset with only the core columns.

    This is a thin wrapper around `load_full_dataset()` that selects
    a reduced set of columns for quick use:
      - 'name'   : molecule’s English name
      - 'smiles' : SMILES string (text encoding of molecular structure)
      - 'MW'     : molecular weight (grams per mole, g/mol)
      - 'bp_k'   : boiling point (Kelvin)

    Returns
    -------
    pandas.DataFrame
        A dataframe containing only the four columns listed above.
    """
    df = load_full_dataset()
    return df[['name', 'smiles', 'MW', 'bp_k']]

def load_alkanes():
    """
    Load a small train/test split of the alkane subset from the full dataset.

    This function selects a curated set of **linear alkanes** (C1–C10) from
    the full dataset, splits them into training and testing groups, and
    prepares features/targets for simple regression tasks (e.g., predicting
    boiling point from molecular weight).

    Splits
    ------
    - Training set: methane, ethane, propane, butane, hexane, heptane, octane, nonane
    - Testing set : pentane, decane

    Features
    --------
    Only molecular weight (**MW**) is used as the input feature.
    Boiling point in Kelvin (**bp_k**) is the target.

    Returns
    -------
    X_train : numpy.ndarray, shape (n_train,)
        Molecular weights of the training molecules.
    X_test : numpy.ndarray, shape (n_test,)
        Molecular weights of the testing molecules.
    y_train : numpy.ndarray, shape (n_train,)
        Boiling points (Kelvin) for the training molecules.
    y_test : numpy.ndarray, shape (n_test,)
        Boiling points (Kelvin) for the testing molecules.
    data : pandas.DataFrame
        A dataframe of the training + testing subset with columns:
        ['name', 'smiles', 'bp_k', 'MW'] — useful for interactive visualization.

    Notes
    -----
    - Data is returned as NumPy arrays for pedagogical convenience, making it
      easy to plug into scikit-learn or matplotlib workflows.
    - Only MW → boiling point is modeled here for simplicity; additional
      descriptors are available via `load_full_dataset()`.
    """

    df = load_full_dataset()

    training_names = ['methane','ethane', 'propane','butane', 'hexane', 'heptane', 'octane', 'nonane']
    testing_names = ['pentane', 'decane']
    df_train = df[df['name'].isin(training_names)]
    df_test = df[df['name'].isin(testing_names)]

    feature_cols = ['MW']

    # slice training and testing sets
    X_train = df_train[feature_cols] # Features
    y_train = df_train['bp_k'] # Target

    X_test = df_test[feature_cols] # Features
    y_test = df_test['bp_k']# Target

    # convert to numpy arrays for pedagogical convenience
    X_train = np.array(X_train).flatten()
    y_train = np.array(y_train)

    X_test = np.array(X_test).flatten()
    y_test = np.array(y_test)

    data = df[df['name'].isin(training_names + testing_names)][['name','smiles','bp_k', 'MW']]

    return X_train, X_test, y_train, y_test, data

def load_expanded_alkanes():
  df = load_full_dataset()

  training_names = ['ethane', 'propane','butane', 'pentane', 'hexane', 'heptane', 'octane', 'nonane']
  testing_names = ['methane', 'decane']
  training_names += ['isobutane', 'isopentane', 'neopentane','isohexane', 'neohexane', 'isoheptane', 'neoheptane', 'isooctane'] # isomers

  df_train = df[df['name'].isin(training_names)]
  df_test = df[df['name'].isin(testing_names)]

  feature_cols = ['MW']

  # slice training and testing sets
  X_train = df_train[feature_cols] # Features
  y_train = df_train['bp_k'] # Target

  X_test = df_test[feature_cols] # Features
  y_test = df_test['bp_k']# Target

  # convert to numpy arrays for pedagogical convenience
  X_train = np.array(X_train).flatten()
  y_train = np.array(y_train)

  X_test = np.array(X_test).flatten()
  y_test = np.array(y_test)

  data = df[df['name'].isin(training_names + testing_names)][['name','smiles','bp_k', 'MW']]

  return X_train, X_test, y_train, y_test, data

def load_multilinear_alkanes():
  df = load_full_dataset()

  training_names = ['methane', 'propane','butane', 'hexane', 'heptane', 'octane', 'nonane', 'decane']
  testing_names = ['ethane', 'pentane']
  training_names += ['isobutane', 'isopentane', 'neopentane','isohexane', 'neohexane', 'isoheptane', 'neoheptane', 'isooctane'] # isomers

  df_train = df[df['name'].isin(training_names)]
  df_test = df[df['name'].isin(testing_names)]

  feature_cols = ['MW', 'branching_index']

  # slice training and testing sets
  X_train = df_train[feature_cols] # Features
  y_train = df_train['bp_k'] # Target

  X_test = df_test[feature_cols] # Features
  y_test = df_test['bp_k']# Target

  return X_train, X_test, y_train, y_test
