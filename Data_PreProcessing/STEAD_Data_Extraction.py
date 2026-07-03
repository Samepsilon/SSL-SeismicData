import seisbench
import seisbench.data as sbd
import numpy as np
import pandas as pd
import h5py
import random
import os
from pathlib import Path

#Default Value
BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR.parent / ".seisbench" / "STEAD"
output_path = BASE_DIR.parent / ".seisbench" / "extractedSTEAD"
sampling_rate_default = 250
component_order_default = "ZNE"
n_samples_default = 3000
seed_default = 42
magnitude_default = 2.5
distance_default = 21

def sample_indices(metadata: pd.DataFrame, trace_category: str, n: int, seed: int) -> pd.Index:
    """Return up to *n* random integer-position indices for a given trace_category."""
    mask = metadata["trace_category"] == trace_category
    available = metadata.index[mask]
    if len(available) < n:
        print(
            f"  [WARNING] Only {len(available)} '{trace_category}' samples available "
            f"(requested {n}). Using all of them."
        )
        return available
    rng = random.Random(seed)
    chosen = rng.sample(list(available), n)
    return pd.Index(sorted(chosen))

def build_subset_dataset(stead: sbd.STEAD, indices: pd.Index) -> sbd.STEAD:
    """
    Return a new STEAD copy whose metadata is restricted to *indices*.
    """
    subset = stead.copy()
    mask = stead._metadata.index.isin(indices)
    subset.filter(mask, inplace=True)
    return subset

def save_subset_dataset(existing_dataset,stead):
    metadata_path = output_path / "metadata.csv"
    waveforms_path = output_path / "waveforms.hdf5"

    dataformat = stead.data_format

    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:

        writer.data_format = dataformat

        for idx in range(len(existing_dataset)):
            waveform = existing_dataset.get_waveforms(idx)
            metadata_dict = existing_dataset.metadata.iloc[idx]
            writer.add_trace(waveform, metadata_dict)



def dataBuilder(
    n_samples = n_samples_default,
    seed = seed_default,
    stead_path = input_path ,
    sampling_rate = sampling_rate_default,
    component_order = component_order_default,
    magnitude = magnitude_default,
    distance = distance_default,

    save_dataset = False,
) -> sbd.STEAD:
    """
    Extract *n_samples* noise and *n_samples* earthquake_local traces from
    STEAD,  return a subset object.

    n_samples : int
        Number of traces to draw per class .
    seed : int
        for reproducibility
    stead_path : Path
    sampling_rate : int
    component_order : str
    magnitude : float
        minimum magnitude for the earthquake
    distance : float
        maximum distance from the station
    save_dataset : bool
        True to simCLR the subset dataset as a Seisbench dataset

    return : subset : sbd.STEAD In-memory SeisBench dataset object containing the randomly selected traces.
    """

    stead = sbd.WaveformDataset(stead_path,sampling_rate=sampling_rate,component_order=component_order)

    # Sample indices for noise and earthquake
    print(f"\nSampling {n_samples} noise traces ")
    noise_idx = sample_indices(stead._metadata, "noise", n_samples, seed=seed)

    mask1 = (
            stead.metadata["source_magnitude"] > magnitude
    )
    mask2 = (
            stead.metadata["source_distance_km"] <= distance
    )
    stead_duplicate = stead.copy()
    stead_duplicate.filter(mask1)
    stead_duplicate.filter(mask2)

    print(f"Sampling {n_samples} earthquake_local traces ")
    eq_idx = sample_indices(stead_duplicate._metadata, "earthquake_local", n_samples, seed=seed + 1)

    combined_idx = noise_idx.append(eq_idx)
    print(f"\nTotal selected: {len(combined_idx)} traces")

    # Build in-memory subset dataset
    print("\nBuilding subset dataset object")
    subset = build_subset_dataset(stead, combined_idx)
    print(f"  Subset size: {len(subset)} traces")

    if save_dataset:
        save_subset_dataset(subset,stead)

    subset.metadata.to_csv(output_path / "metadata.csv", index=False)

    return subset



if __name__ == '__main__':
    sample_dataset = dataBuilder()

