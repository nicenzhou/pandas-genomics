from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor
from ...arrays import GenotypeDtype, GenotypeArray
from ...scalars import Variant, MISSING_IDX

def from_plink(
    input: Union[str, Path],
    swap_alleles: bool = False,
    max_variants: Optional[int] = None,
    categorical_phenotype: bool = True,
    num_threads: int = 1,
    max_memory: int = None,
):
    def load_genotypes_parallel(bed_file, variant_list, num_samples, swap_alleles):
        chunk_size = num_samples // 4
        if num_samples % 4 > 0:
            chunk_size += 1

        def load_genotype_for_variant(v_idx):
            variant = variant_list[v_idx]
            with open(bed_file, 'rb') as file:
                file.seek(3 + v_idx * chunk_size)
                variant_gt_bytes = np.frombuffer(file.read(chunk_size), dtype='uint8')
            gt_array = create_gt_array(num_samples, variant_gt_bytes, variant)
            if swap_alleles:
                gt_array.set_reference(1)
            return (f"{v_idx}_{gt_array.variant.id}", gt_array)

        # Limit memory usage if max_memory is specified
        if max_memory:
            max_memory_bytes = max_memory * 1024 * 1024
            per_thread_memory = max_memory_bytes / num_threads
            chunk_size = min(chunk_size, per_thread_memory)

        # Use ThreadPoolExecutor to load genotypes in parallel with user-specified threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            gt_arrays = list(executor.map(load_genotype_for_variant, range(len(variant_list)))

        return dict(gt_arrays)

    input = str(input)
    bed_file = Path(input + ".bed")
    bim_file = Path(input + ".bim")
    fam_file = Path(input + ".fam")

    if not bed_file.exists() or not bim_file.exists() or not fam_file.exists():
        raise ValueError("One or more PLINK files not found.")

    print(f"Loading genetic data from '{bed_file.stem}'")

    # Load fam file
    df = load_sample_info(fam_file, categorical_phenotype)
    # Load bim file
    variant_list = load_variant_info(bim_file, max_variants)
    # Load bed file using multiple threads and memory limit
    gt_array_dict = load_genotypes_parallel(bed_file, variant_list, len(df), swap_alleles)

    # Merge with sample allele index
    df = pd.concat([df, pd.DataFrame.from_dict(gt_array_dict)], axis=1)
    df = df.set_index(["FID", "IID", "IID_father", "IID_mother", "sex", "phenotype"])

    return df

def load_variant_info(bim_file, max_variants):
    variant_info = pd.read_table(bim_file, header=None, sep="\t")
    variant_info.columns = [
        "chromosome",
        "variant_id",
        "position",
        "coordinate",
        "allele1",
        "allele2",
    ]
    variant_info["chromosome"] = variant_info["chromosome"].astype("category")
    if max_variants is not None:
        if max_variants < 1:
            raise ValueError(f"'max_variants' set to an invalid value: {max_variants}")
        else:
            variant_info = variant_info.iloc[:max_variants]
    variant_list = [create_variant(row) for idx, row in variant_info.iterrows()]
    return variant_list

def create_variant(variant_info_row):
    variant_id = str(variant_info_row["variant_id"])
    a1 = str(variant_info_row["allele1"])
    a2 = str(variant_info_row["allele2"])
    if a2 == "0":
        a2 = None
    if a1 == "0":
        a1 = None
    else:
        a1 = [a1]
    if np.isnan(variant_info_row["chromosome"]):
        chromosome = None
    else:
        chromosome = str(variant_info_row["chromosome"])
    variant = Variant(
        chromosome=chromosome,
        position=int(variant_info_row["coordinate"]),
        id=variant_id,
        ref=a2,
        alt=a1,
        ploidy=2,
    )
    return variant

def load_sample_info(fam_file, categorical_phenotype):
    df = pd.read_table(fam_file, header=None, sep=" ")
    df.columns = ["FID", "IID", "IID_father", "IID_mother", "sex", "phenotype"]
    df["sex"] = df["sex"].astype("category")
    df["sex"] = df["sex"].cat.rename_categories({1: "male", 2: "female", 0: "unknown"})
    DEFAULT_CAT_MAP = {1: "Control", 2: "Case"}
    if categorical_phenotype:
        df["phenotype"] = df["phenotype"].astype("category")
        df["phenotype"].cat.rename_categories(DEFAULT_CAT_MAP, inplace=True)
        df.loc[~df["phenotype"].isin(DEFAULT_CAT_MAP.values()), "phenotype"] = None
    print(f"\tLoaded information for {len(df)} samples from '{fam_file.name}'")
    return df

def create_gt_array(num_samples, variant_gt_bytes, variant):
    genotypes = np.flip(np.unpackbits(variant_gt_bytes).reshape(-1, 4, 2), axis=1)
    genotypes = genotypes.reshape(-1, 2)[:num_samples]
    missing_gt = (genotypes == (0, 1)).all(axis=1)
    genotypes[missing_gt] = (MISSING_IDX, MISSING_IDX)
    het_gt = (genotypes == (1, 0)).all(axis=1)
    genotypes[het_gt] = (0, 1)
    dtype = GenotypeDtype(variant)
    scores = np.ones(num_samples) * MISSING_IDX
    data = np.array(list(zip(genotypes, scores)), dtype=dtype._record_type)
    gt_array = GenotypeArray(values=data, dtype=dtype)
    return gt_array

# Example usage with user-specified threads and memory:
# df = from_plink('your_input_path', swap_alleles=True, max_variants=None, categorical_phenotype=True, num_threads=4, max_memory=4096)
