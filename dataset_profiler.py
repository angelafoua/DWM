#!/usr/bin/env python
"""
Dataset Profiler for DWM Empirical Study (Paper 1).

Computes all independent variables (dataset characteristics) from idea.txt:
  - Size features: num records, num attributes, avg record length
  - Token features: unique tokens, freq mean/std/skewness/kurtosis
  - Information features: dataset entropy, attribute entropy, avg token entropy
  - Quality features: missing values %, duplicate density, noise level
  - Similarity features: mean similarity, similarity variance (sampled pairs)

Usage:
    python dataset_profiler.py S7GX.txt --truth truthABCgoodDQ.txt
    python dataset_profiler.py S7GX.txt --truth truthABCgoodDQ.txt --output profiles.csv
    python dataset_profiler.py --batch batch_datasets.txt --output profiles.csv

batch_datasets.txt format (one per line):
    S7GX.txt,truthABCgoodDQ.txt
    S8P.txt,truthABCpoorDQ.txt
"""

import argparse
import csv
import math
import os
import re
import sys
import random
from collections import Counter

import numpy as np
from scipy import stats


PROFILE_COLUMNS = [
    "dataset",
    "num_records",
    "num_attributes",
    "avg_record_length",
    "total_tokens",
    "unique_tokens",
    "unique_token_ratio",
    "token_freq_min",
    "token_freq_max",
    "token_freq_mean",
    "token_freq_std",
    "token_freq_skewness",
    "token_freq_kurtosis",
    "numeric_token_count",
    "numeric_token_ratio",
    "token_len_min",
    "token_len_max",
    "token_len_avg",
    "token_len_std",
    "dataset_entropy",
    "avg_attribute_entropy",
    "avg_token_entropy",
    "missing_values_pct",
    "duplicate_density",
    "mean_similarity",
    "similarity_variance",
]


def tokenize(body):
    text = body.upper()
    text = re.sub(r"[\W]+", " ", text)
    return [t for t in text.split() if t]


def read_dataset(filepath, delimiter=",", has_header=True):
    """Read a DWM-format CSV and return raw rows, field counts, and refDict."""
    records = []
    ref_dict = {}
    num_attributes = 0

    with open(filepath, "r", errors="replace") as fh:
        if has_header:
            header = fh.readline()
            if header.strip():
                num_attributes = len(header.strip().split(delimiter)) - 1
        for line in fh:
            line = line.strip()
            if not line:
                continue
            idx = line.find(delimiter)
            if idx < 0:
                continue
            ref_id = line[:idx]
            body = line[idx + 1 :]
            fields = body.split(delimiter)
            if num_attributes == 0:
                num_attributes = len(fields)
            records.append((ref_id, fields, body))
            ref_dict[ref_id] = tokenize(body)
    return records, ref_dict, num_attributes


def read_truth(filepath):
    """Read a truth file and return dict: recID -> truthID."""
    truth = {}
    with open(filepath, "r") as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                truth[parts[0].strip()] = parts[1].strip()
    return truth


def compute_size_features(records, ref_dict, num_attributes):
    num_records = len(records)
    token_counts = [len(toks) for toks in ref_dict.values()]
    avg_record_length = np.mean(token_counts) if token_counts else 0.0
    return num_records, num_attributes, round(avg_record_length, 4)


def compute_token_features(ref_dict):
    freq = Counter()
    token_len_counts = Counter()
    total = 0
    numeric_count = 0
    for tokens in ref_dict.values():
        for t in tokens:
            freq[t] += 1
            token_len_counts[len(t)] += 1
            total += 1
            if t.isdigit():
                numeric_count += 1

    unique = len(freq)
    unique_ratio = round(unique / total, 4) if total else 0.0
    numeric_ratio = round(numeric_count / total, 4) if total else 0.0

    freqs = np.array(list(freq.values()), dtype=float)
    freq_min = int(np.min(freqs)) if len(freqs) > 0 else 0
    freq_max = int(np.max(freqs)) if len(freqs) > 0 else 0
    freq_mean = round(float(np.mean(freqs)), 4)
    freq_std = round(float(np.std(freqs, ddof=1)), 4) if len(freqs) > 1 else 0.0
    freq_skew = round(float(stats.skew(freqs, bias=False)), 4) if len(freqs) > 2 else 0.0
    freq_kurt = round(float(stats.kurtosis(freqs, bias=False)), 4) if len(freqs) > 3 else 0.0

    # Token length statistics (weighted by frequency, matching DWM16 approach)
    len_min = min(token_len_counts.keys()) if token_len_counts else 0
    len_max = max(token_len_counts.keys()) if token_len_counts else 0
    total_f = sum(token_len_counts.values())
    total_fxl = sum(f * l for l, f in token_len_counts.items())
    total_fxl2 = sum(f * l * l for l, f in token_len_counts.items())
    len_avg = round(total_fxl / total_f, 4) if total_f > 0 else 0.0
    if total_f > 1:
        variance = (total_f * total_fxl2 - total_fxl * total_fxl) / (total_f * (total_f - 1))
        len_std = round(math.sqrt(max(variance, 0)), 4)
    else:
        len_std = 0.0

    return (
        total,
        unique,
        unique_ratio,
        freq_min,
        freq_max,
        freq_mean,
        freq_std,
        freq_skew,
        freq_kurt,
        numeric_count,
        numeric_ratio,
        len_min,
        len_max,
        len_avg,
        len_std,
        freq,
    )


def compute_entropy_features(records, ref_dict, token_freq, num_attributes):
    total_tokens = sum(token_freq.values())

    # Dataset-level entropy: Shannon entropy of the token frequency distribution
    dataset_entropy = 0.0
    for count in token_freq.values():
        p = count / total_tokens
        if p > 0:
            dataset_entropy -= p * math.log2(p)
    dataset_entropy = round(dataset_entropy, 4)

    # Per-attribute entropy: split each record's fields, compute entropy per column
    if num_attributes > 0:
        attr_values = [[] for _ in range(num_attributes)]
        for _, fields, _ in records:
            for i in range(min(len(fields), num_attributes)):
                val = fields[i].strip().upper()
                attr_values[i].append(val)

        attr_entropies = []
        for col_vals in attr_values:
            counts = Counter(col_vals)
            n = len(col_vals)
            if n == 0:
                continue
            ent = 0.0
            for c in counts.values():
                p = c / n
                if p > 0:
                    ent -= p * math.log2(p)
            attr_entropies.append(ent)
        avg_attr_entropy = round(np.mean(attr_entropies), 4) if attr_entropies else 0.0
    else:
        avg_attr_entropy = 0.0

    # Avg token entropy: average information content per token
    avg_token_entropy = 0.0
    if total_tokens > 0:
        for count in token_freq.values():
            p = count / total_tokens
            avg_token_entropy += -math.log2(p)
        avg_token_entropy = round(avg_token_entropy / len(token_freq), 4)

    return dataset_entropy, avg_attr_entropy, avg_token_entropy


def compute_quality_features(records, num_attributes, truth_dict, ref_dict):
    # Missing values %
    total_fields = 0
    missing_fields = 0
    for _, fields, _ in records:
        for f in fields:
            total_fields += 1
            if f.strip() == "":
                missing_fields += 1
    missing_pct = round(missing_fields / total_fields * 100, 2) if total_fields else 0.0

    # Duplicate density: fraction of records that are duplicates (have a truth group > 1)
    dup_density = 0.0
    if truth_dict:
        group_counts = Counter(truth_dict.values())
        dup_records = sum(c for c in group_counts.values() if c > 1)
        matched_records = sum(1 for rid in ref_dict if rid in truth_dict)
        if matched_records > 0:
            dup_density = round(dup_records / matched_records, 4)

    return missing_pct, dup_density


def jaccard_similarity(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_similarity_features(ref_dict, sample_size=10000, seed=42):
    ref_ids = list(ref_dict.keys())
    n = len(ref_ids)
    if n < 2:
        return 0.0, 0.0

    rng = random.Random(seed)
    max_pairs = n * (n - 1) // 2
    actual_sample = min(sample_size, max_pairs)

    similarities = []
    seen = set()
    attempts = 0
    max_attempts = actual_sample * 10
    while len(similarities) < actual_sample and attempts < max_attempts:
        attempts += 1
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)
        sim = jaccard_similarity(ref_dict[ref_ids[i]], ref_dict[ref_ids[j]])
        similarities.append(sim)

    if not similarities:
        return 0.0, 0.0

    arr = np.array(similarities)
    return round(float(np.mean(arr)), 4), round(float(np.var(arr)), 4)


def profile_dataset(data_file, truth_file=None, delimiter=",", has_header=True):
    records, ref_dict, num_attributes = read_dataset(data_file, delimiter, has_header)

    truth_dict = {}
    if truth_file and os.path.exists(truth_file):
        truth_dict = read_truth(truth_file)

    num_records, num_attrs, avg_rec_len = compute_size_features(records, ref_dict, num_attributes)

    (
        total_tokens, unique_tokens, unique_ratio,
        freq_min, freq_max, freq_mean, freq_std, freq_skew, freq_kurt,
        numeric_count, numeric_ratio,
        len_min, len_max, len_avg, len_std,
        token_freq,
    ) = compute_token_features(ref_dict)

    dataset_ent, avg_attr_ent, avg_tok_ent = compute_entropy_features(
        records, ref_dict, token_freq, num_attributes
    )

    missing_pct, dup_density = compute_quality_features(
        records, num_attributes, truth_dict, ref_dict
    )

    mean_sim, sim_var = compute_similarity_features(ref_dict)

    profile = {
        "dataset": os.path.basename(data_file),
        "num_records": num_records,
        "num_attributes": num_attrs,
        "avg_record_length": avg_rec_len,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "unique_token_ratio": unique_ratio,
        "token_freq_min": freq_min,
        "token_freq_max": freq_max,
        "token_freq_mean": freq_mean,
        "token_freq_std": freq_std,
        "token_freq_skewness": freq_skew,
        "token_freq_kurtosis": freq_kurt,
        "numeric_token_count": numeric_count,
        "numeric_token_ratio": numeric_ratio,
        "token_len_min": len_min,
        "token_len_max": len_max,
        "token_len_avg": len_avg,
        "token_len_std": len_std,
        "dataset_entropy": dataset_ent,
        "avg_attribute_entropy": avg_attr_ent,
        "avg_token_entropy": avg_tok_ent,
        "missing_values_pct": missing_pct,
        "duplicate_density": dup_density,
        "mean_similarity": mean_sim,
        "similarity_variance": sim_var,
    }
    return profile


def print_profile(profile):
    print(f"\n{'='*60}")
    print(f"  Dataset Profile: {profile['dataset']}")
    print(f"{'='*60}")
    print(f"\n  SIZE FEATURES")
    print(f"    Records:             {profile['num_records']}")
    print(f"    Attributes:          {profile['num_attributes']}")
    print(f"    Avg Record Length:   {profile['avg_record_length']} tokens")
    print(f"\n  TOKEN FEATURES")
    print(f"    Total Tokens:        {profile['total_tokens']}")
    print(f"    Unique Tokens:       {profile['unique_tokens']}")
    print(f"    Unique Ratio:        {profile['unique_token_ratio']}")
    print(f"    Freq Min:            {profile['token_freq_min']}")
    print(f"    Freq Max:            {profile['token_freq_max']}")
    print(f"    Freq Mean:           {profile['token_freq_mean']}")
    print(f"    Freq Std Dev:        {profile['token_freq_std']}")
    print(f"    Freq Skewness:       {profile['token_freq_skewness']}")
    print(f"    Freq Kurtosis:       {profile['token_freq_kurtosis']}")
    print(f"    Numeric Tokens:      {profile['numeric_token_count']}")
    print(f"    Numeric Ratio:       {profile['numeric_token_ratio']}")
    print(f"    Token Len Min:       {profile['token_len_min']}")
    print(f"    Token Len Max:       {profile['token_len_max']}")
    print(f"    Token Len Avg:       {profile['token_len_avg']}")
    print(f"    Token Len Std Dev:   {profile['token_len_std']}")
    print(f"\n  INFORMATION FEATURES")
    print(f"    Dataset Entropy:     {profile['dataset_entropy']}")
    print(f"    Avg Attr Entropy:    {profile['avg_attribute_entropy']}")
    print(f"    Avg Token Entropy:   {profile['avg_token_entropy']}")
    print(f"\n  QUALITY FEATURES")
    print(f"    Missing Values %:    {profile['missing_values_pct']}")
    print(f"    Duplicate Density:   {profile['duplicate_density']}")
    print(f"\n  SIMILARITY FEATURES (sampled 10K pairs)")
    print(f"    Mean Similarity:     {profile['mean_similarity']}")
    print(f"    Similarity Variance: {profile['similarity_variance']}")
    print()


def write_csv(profiles, output_file):
    with open(output_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        for p in profiles:
            writer.writerow(p)
    print(f"Profiles written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Profiler for DWM Study")
    parser.add_argument("datafile", nargs="?", help="Input dataset CSV file")
    parser.add_argument("--truth", default=None, help="Truth file for duplicate density")
    parser.add_argument("--delimiter", default=",", help="Field delimiter (default: comma)")
    parser.add_argument("--no-header", action="store_true", help="Input has no header row")
    parser.add_argument("--output", default=None, help="Output CSV file for profiles")
    parser.add_argument(
        "--batch", default=None,
        help="Batch file: each line is datafile,truthfile",
    )
    args = parser.parse_args()

    profiles = []

    if args.batch:
        with open(args.batch, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",", 1)
                data_f = parts[0].strip()
                truth_f = parts[1].strip() if len(parts) > 1 else None
                print(f"Profiling {data_f} ...")
                p = profile_dataset(
                    data_f, truth_f, args.delimiter, not args.no_header
                )
                print_profile(p)
                profiles.append(p)
    elif args.datafile:
        p = profile_dataset(
            args.datafile, args.truth, args.delimiter, not args.no_header
        )
        print_profile(p)
        profiles.append(p)
    else:
        parser.print_help()
        sys.exit(1)

    if args.output and profiles:
        write_csv(profiles, args.output)


if __name__ == "__main__":
    main()
