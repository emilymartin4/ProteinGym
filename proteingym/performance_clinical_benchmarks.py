"""Clinical benchmark performance script for ProteinGym.

Computes AUC metrics for clinical variant pathogenicity prediction:
- Substitutions: AUC computed per-gene, then averaged
- Indels: All genes pooled together, then single AUC computed
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_bootstrap_standard_error(auc_per_gene, n_bootstrap=10000):
    """Compute bootstrap standard error for mean AUC across genes."""
    aucs = auc_per_gene.values
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(aucs, size=len(aucs), replace=True)
        bootstrap_means.append(np.nanmean(sample))
    return np.std(bootstrap_means, ddof=1)


def main():
    parser = argparse.ArgumentParser(description="ProteinGym clinical benchmark performance")
    parser.add_argument("--input_scoring_files_folder", type=str, required=True, help="Folder containing model score files (one per gene)")
    parser.add_argument("--output_performance_file_folder", type=str, required=True, help="Folder to save performance results")
    parser.add_argument("--clinical_reference_file_path", type=str, required=True, help="Reference file with list of clinical genes")
    parser.add_argument("--indel_mode", action="store_true", help="Use indel mode: pool all variants then compute AUC")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"),
        help="Path to config.json",
    )
    args = parser.parse_args()

    # Load reference file and config
    reference = pd.read_csv(args.clinical_reference_file_path)
    gene_ids = reference["DMS_id"].tolist()
    print(f"Found {len(gene_ids)} genes in reference file")

    proteingym_folder = os.path.dirname(os.path.realpath(__file__))
    with open(args.config_file) as f:
        config = json.load(f)
    with open(os.path.join(proteingym_folder, "constants.json")) as f:
        constants = json.load(f)

    clean_names = constants.get("clean_names", {})

    # Get model list based on mode
    if args.indel_mode:
        model_config = config["model_list_zero_shot_indels_clinical"]
    else:
        model_config = config["model_list_zero_shot_substitutions_clinical"]

    model_names = list(model_config.keys())
    print(f"Evaluating {len(model_names)} models")

    # Create output directory structure
    mutation_type = "indels" if args.indel_mode else "substitutions"
    output_dir = os.path.join(args.output_performance_file_folder, mutation_type, "AUC")
    os.makedirs(output_dir, exist_ok=True)

    if args.indel_mode:
        # Indels: pool all variants across genes, compute single AUC
        summary_df = compute_pooled_auc(
            gene_ids,
            model_names,
            model_config,
            args.input_scoring_files_folder,
            clean_names,
        )
    else:
        # Substitutions: compute AUC per gene, then average
        summary_df, dms_level_df = compute_per_gene_auc(
            gene_ids,
            model_names,
            model_config,
            args.input_scoring_files_folder,
            clean_names,
        )
        # Save DMS-level results
        dms_level_path = os.path.join(output_dir, f"clinical_{mutation_type}_AUC_DMS_level.csv")
        dms_level_df.to_csv(dms_level_path, index=False)
        print(f"DMS-level results saved to {dms_level_path}")

    # Save summary results
    summary_path = os.path.join(output_dir, f"Summary_performance_clinical_{mutation_type}_AUC.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary results saved to {summary_path}")
    print(summary_df.to_string(index=False))


def compute_per_gene_auc(gene_ids, model_names, model_config, scores_folder, clean_names):
    """Compute AUC per gene for each model, then average across genes.

    Returns:
        summary_df: DataFrame with average AUC and bootstrap SE for each model
        dms_level_df: DataFrame with per-gene AUCs

    """
    # Store per-gene AUCs for each model
    gene_aucs = {model: {} for model in model_names}

    for gene_id in gene_ids:
        score_file = os.path.join(scores_folder, f"{gene_id}.csv")
        if not os.path.exists(score_file):
            print(f"Missing score file for {gene_id}")
            continue

        df = pd.read_csv(score_file)

        if "DMS_bin_score" not in df.columns:
            print(f"Missing DMS_bin_score column for {gene_id}")
            continue

        y_true = df["DMS_bin_score"].values
        # convert to binary labels if there are strings
        y_true = [1 if val == "Pathogenic" else val for val in y_true]
        y_true = [0 if val == "Benign" else val for val in y_true]
        # flip labels (since pathogenic should be less fit)
        y_true = np.array(y_true)
        assert np.all((y_true == 1) | (y_true == 0)), "Labels contain NaNs"
        y_true = 1 - y_true

        # # Skip if no variance in labels
        if len(np.unique(y_true[~np.isnan(y_true)])) < 2:
            print(f"Skipping {gene_id}: insufficient label variance")
            continue

        for model in model_names:
            if model not in df.columns:
                continue
            y_score = df[model].values
            directionality = model_config[model]["directionality"]
            y_score = y_score * directionality

            valid_mask = ~np.isnan(y_score)

            if valid_mask.sum() == 0:
                continue
            if len(np.unique(y_true[valid_mask])) < 2:
                print(f"Skipping AUC for {model} on {gene_id}: insufficient label variance")
                continue

            try:
                auc = roc_auc_score(y_true[valid_mask], y_score[valid_mask])
                gene_aucs[model][gene_id] = round(auc, 3)
            except ValueError as e:
                print(f"AUC error for {model} on {gene_id}: {e}")

    # Build DMS-level DataFrame
    all_gene_ids = sorted(set().union(*[set(gene_aucs[m].keys()) for m in model_names]))
    dms_level_data = {"RefSeq ID": all_gene_ids}
    for model in model_names:
        model_clean = clean_names.get(model, model)
        dms_level_data[model_clean] = [gene_aucs[model].get(g, np.nan) for g in all_gene_ids]
    dms_level_df = pd.DataFrame(dms_level_data)

    # Compute summary statistics
    results = []
    for model in model_names:
        aucs = pd.Series(gene_aucs[model])
        model_clean = clean_names.get(model, model)
        if len(aucs) == 0:
            results.append(
                {
                    "Model_rank": None,
                    "Model_name": model_clean,
                    "Model type": model_config[model]["model_type"],
                    "Average_AUC": np.nan,
                    "Bootstrap_standard_error_AUC": np.nan,
                    "Num_genes": 0,
                },
            )
        else:
            results.append(
                {
                    "Model_rank": None,
                    "Model_name": model_clean,
                    "Model type": model_config[model]["model_type"],
                    "Average_AUC": round(aucs.mean(), 3),
                    "Bootstrap_standard_error_AUC": round(compute_bootstrap_standard_error(aucs), 3),
                    "Num_genes": len(aucs),
                },
            )

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values("Average_AUC", ascending=False).reset_index(drop=True)
    summary_df["Model_rank"] = summary_df.index + 1

    return summary_df, dms_level_df


def compute_pooled_auc(gene_ids, model_names, model_config, scores_folder, clean_names):
    """Pool all variants across genes, then compute a single AUC per model.
    Used for indels where per-gene sample sizes are too small.
    """
    all_data = []
    for gene_id in gene_ids:
        score_file = os.path.join(scores_folder, f"{gene_id}.csv")
        if not os.path.exists(score_file):
            print(f"Missing score file for {gene_id}")
            continue

        df = pd.read_csv(score_file)
        if "DMS_score_bin" not in df.columns:
            print(f"Missing DMS_score_bin column for {gene_id}")
            continue

        all_data.append(df)

    if not all_data:
        print("No data loaded!")
        return pd.DataFrame()

    pooled_df = pd.concat(all_data, ignore_index=True)
    print(f"Pooled {len(pooled_df)} variants across {len(all_data)} genes")

    y_true = pooled_df["DMS_score_bin"].values
    # convert to binary labels if there are strings
    y_true = [1 if val == "Pathogenic" else val for val in y_true]
    y_true = [0 if val == "Benign" else val for val in y_true]
    # flip labels (since pathogenic should be less fit)
    y_true = np.array(y_true)
    assert np.all((y_true == 1) | (y_true == 0)), "Labels contain NaNs"
    y_true = 1 - y_true
    results = []
    for model in model_names:
        model_clean = clean_names.get(model, model)
        if model not in pooled_df.columns:
            results.append(
                {
                    "Model_rank": None,
                    "Model_name": model_clean,
                    "Model type": model_config[model]["model_type"],
                    "Average_AUC": np.nan,
                    "Num_variants": 0,
                },
            )
            continue

        y_score = pooled_df[model].values
        directionality = model_config[model]["directionality"]
        y_score = y_score * directionality
        valid_mask = ~np.isnan(y_score)

        try:
            auc = roc_auc_score(y_true[valid_mask], y_score[valid_mask])
            results.append(
                {
                    "Model_rank": None,
                    "Model_name": model_clean,
                    "Model type": model_config[model]["model_type"],
                    "Average_AUC": round(auc, 3),
                    "Num_variants": int(valid_mask.sum()),
                },
            )
        except ValueError as e:
            print(f"AUC error for {model}: {e}")
            results.append(
                {
                    "Model_rank": None,
                    "Model_name": model_clean,
                    "Model type": model_config[model]["model_type"],
                    "Average_AUC": np.nan,
                    "Num_variants": 0,
                },
            )

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values("Average_AUC", ascending=False).reset_index(drop=True)
    summary_df["Model_rank"] = summary_df.index + 1

    return summary_df


if __name__ == "__main__":
    main()
