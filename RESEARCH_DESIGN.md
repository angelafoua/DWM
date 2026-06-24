# An Empirical Study of Dataset Characteristics and Parameter Sensitivity in the Data Washing Machine

## Research Objective

Determine which dataset characteristics have the greatest influence on DWM parameter selection and entity resolution (ER) performance. This is a pure empirical study: not prediction, not machine learning, just understanding the relationships.

## Research Questions

### RQ1: How sensitive is DWM performance to changes in each parameter?

Parameters under study:

| Parameter | Role | Type |
|-----------|------|------|
| beta      | Maximum frequency of a blocking token | Integer (>= 2) |
| sigma     | Stop word threshold (tokens with freq > sigma removed before matching) | Integer (> beta) |
| mu        | Match threshold to link two references | Float (0.0, 1.0] |
| epsilon   | Minimum cluster quality to keep each iteration | Float (0.0, 1.0] |

### RQ2: Which dataset characteristics correlate with optimal parameter values?

### RQ3: Are some parameters more dataset-dependent than others?

For example, beta might vary greatly across datasets while epsilon might remain stable.

## Variables

### Independent Variables (Dataset Characteristics)

Computed by `dataset_profiler.py`.

#### Size Features

| Feature | Description |
|---------|-------------|
| num_records | Number of entity references in the dataset |
| num_attributes | Number of fields/columns (excluding refID) |
| avg_record_length | Average number of tokens per record |

#### Token Features

| Feature | Description |
|---------|-------------|
| total_tokens | Total token count across all records |
| unique_tokens | Number of distinct tokens |
| unique_token_ratio | unique_tokens / total_tokens |
| token_freq_min | Minimum token frequency |
| token_freq_max | Maximum token frequency |
| token_freq_mean | Mean token frequency |
| token_freq_std | Standard deviation of token frequencies |
| token_freq_skewness | Skewness of the token frequency distribution |
| token_freq_kurtosis | Kurtosis of the token frequency distribution |
| numeric_token_count | Number of all-digit tokens |
| numeric_token_ratio | Fraction of tokens that are all-digit |
| token_len_min | Minimum token length |
| token_len_max | Maximum token length |
| token_len_avg | Average token length (frequency-weighted) |
| token_len_std | Standard deviation of token length (frequency-weighted) |

#### Information Features

| Feature | Description |
|---------|-------------|
| dataset_entropy | Shannon entropy of the token frequency distribution |
| avg_attribute_entropy | Average Shannon entropy across all attribute columns |
| avg_token_entropy | Average information content per unique token |

#### Quality Features

| Feature | Description |
|---------|-------------|
| missing_values_pct | Percentage of empty fields across all records |
| duplicate_density | Fraction of records belonging to duplicate groups (requires truth file) |

#### Similarity Features

| Feature | Description |
|---------|-------------|
| mean_similarity | Mean Jaccard similarity from 10,000 sampled record pairs |
| similarity_variance | Variance of Jaccard similarity from 10,000 sampled record pairs |

### Dependent Variables

#### Optimal Parameters (per dataset)

The parameter configuration that maximizes F1:

- beta* (optimal beta)
- sigma* (optimal sigma)
- mu* (optimal mu)
- epsilon* (optimal epsilon)

#### Performance Metrics (per configuration)

| Metric | Description |
|--------|-------------|
| Precision | TP / Linked Pairs |
| Recall | TP / Expected Pairs |
| F1 | Harmonic mean of Precision and Recall |

## Datasets

Target: 20-50 datasets minimum.

### Synthetic Datasets (available in repo)

- S7PX, S8PX variants with truthABCgoodDQ / truthABCpoorDQ
- S12PX, S14GX, S16PX
- S1G, S2G, S4G, S5G
- SOG-generated datasets with controlled noise, duplicates, and missing values

### Real Datasets

- S3Rest (Restaurant dataset)
- S6GeCo (GeCo-generated dataset)
- FEBRL (Freely Extensible Biomedical Record Linkage)
- Other publicly available ER benchmarks

## Experimental Procedure

### Step 1: Profile Each Dataset

```bash
python dataset_profiler.py --batch batch_datasets.txt --output dataset_profiles.csv
```

Produces a CSV row per dataset with all 26 independent variable measurements.

### Step 2: Parameter Sweep

```bash
python parameter_sweep.py --batch batch_sweep.txt \
    --beta 2,5,10,15,20,30,50 \
    --sigma 6,12,25,50,100,200 \
    --mu 0.50:0.95:0.05 \
    --epsilon 0.05:0.50:0.05 \
    --output sweep_results.csv
```

For each dataset, tests all valid combinations of (beta, sigma, mu, epsilon) and records Precision, Recall, F1, plus token statistics for every configuration.

### Step 3: Identify Optimal Configurations

For each dataset, extract the parameter combination that maximizes F1.

| Dataset | beta* | sigma* | mu* | epsilon* | Best F1 |
|---------|-------|--------|-----|----------|---------|
| S7GX    | ?     | ?      | ?   | ?        | ?       |
| S8P     | ?     | ?      | ?   | ?        | ?       |
| ...     | ...   | ...    | ... | ...      | ...     |

### Step 4: Sensitivity Analysis

For each parameter, compute:

```
Sensitivity = delta_F1 / delta_Parameter
```

How much F1 changes when a single parameter changes while others are held at their optimal values.

| Parameter | Avg Sensitivity | Interpretation |
|-----------|----------------|----------------|
| beta      | ?              | ?              |
| sigma     | ?              | ?              |
| mu        | ?              | ?              |
| epsilon   | ?              | ?              |

### Step 5: Statistical Analysis

#### Correlation Analysis

Compute Pearson and Spearman correlations between each dataset characteristic and each optimal parameter value.

| Feature | Corr with beta* | Corr with sigma* | Corr with mu* | Corr with epsilon* |
|---------|-----------------|-------------------|---------------|---------------------|
| dataset_entropy | ? | ? | ? | ? |
| duplicate_density | ? | ? | ? | ? |
| token_freq_std | ? | ? | ? | ? |
| ... | ... | ... | ... | ... |

#### Feature Importance (Regression)

Fit simple models (Linear Regression, Random Forest) from dataset features to each optimal parameter. Not for prediction, but for feature importance ranking.

| Feature | Importance for mu* | Importance for beta* |
|---------|-------------------|---------------------|
| dataset_entropy | ? | ? |
| token_freq_std | ? | ? |
| duplicate_density | ? | ? |

## Expected Contributions

1. First systematic parameter sensitivity study of DWM
2. Identification of which dataset characteristics most affect DWM behavior
3. Evidence for which parameters should be automatically configured vs. left at defaults
4. Foundation for future AutoDWM research (predictive parameter selection)

## Tools

| Script | Purpose |
|--------|---------|
| `dataset_profiler.py` | Compute all dataset characteristics (independent variables) |
| `parameter_sweep.py` | Automate grid search over DWM parameters and collect results |
| `DWM00_Driver.py` | Original DWM driver (for validation) |

## Why This Design Works

This is a low-risk study design. Even if parameter prediction fails in future work, this paper still publishes: "These dataset properties influence DWM behavior." That is a legitimate scientific result. The study does not need to prove automatic tuning is possible -- it only needs to prove that measurable dataset characteristics are associated with parameter behavior.
