import pandas as pd
import numpy as np
from aucopt.optim.alm import run_alm
from numpy.linalg import norm
from numpy import dot, arccos
import glob
import os
from math import acos, degrees

def summarize_almlog(almlog, dataset_name):
    """
    Summarizes ALM optimization log statistics into a single-row DataFrame.

    Parameters:
    - almlog: ALMLog object
    - dataset_name: string name for dataset

    Returns:
    - df: a one-row pandas DataFrame with summary statistics
    """

    T = almlog.alm_iter  # Number of ALM iterations actually run

    # Handle edge case: no ALM iterations
    if T == 0:
        return pd.DataFrame([{
            "dataset": dataset_name,
            "alm_iter": 0,
            "L_final": almlog.L_final,
            "alm_time": almlog.alm_time,
            "ssn_iters_total": 0,
            "ssn_iters_avg": 0.0,
            "ssn_time_total": 0.0,
            "ssn_time_avg": 0.0,
            "ssn_wD_time_total": 0.0,
            "ssn_wD_time_avg": 0.0,
            "prox_time_total": 0.0,
            "prox_time_avg": 0.0,
            "lsearch_time_total": 0.0,
            "lsearch_time_avg": 0.0,
            "lsearch_iters_total": 0,
            "lsearch_iters_avg": 0.0,
            "prox_allocs_total": 0,
            "prox_allocs_avg": 0.0,
            "lsearch_allocs_total": 0,
            "lsearch_allocs_avg": 0.0
        }])

    # Slice arrays to actual iterations
    ssn_iters_used = almlog.ssn_iters[:T]
    ssn_times = almlog.ssn_times[:T]
    ssn_wD_times = almlog.ssn_wD_times[:T]
    prox_times = almlog.prox_times[:T, :].flatten()
    lsearch_times = almlog.lsearch_times[:T, :, :].flatten()
    lsearch_iters = almlog.lsearch_iters[:T, :].flatten()
    prox_allocs = almlog.prox_allocs[:T]
    lsearch_allocs = almlog.lsearch_allocs[:T]

    total_ssn_iters = int(np.sum(ssn_iters_used))
    total_lsearch_iters = int(np.sum(lsearch_iters))

    summary = {
        "dataset": dataset_name,
        "alm_iter": T,
        "L_final": almlog.L_final,
        "alm_time": almlog.alm_time,

        "ssn_iters_total": total_ssn_iters,
        "ssn_iters_avg": np.mean(ssn_iters_used) if ssn_iters_used.size > 0 else 0.0,

        "ssn_time_total": np.sum(ssn_times),
        "ssn_time_avg": np.mean(ssn_times) if ssn_times.size > 0 else 0.0,

        "ssn_wD_time_total": np.sum(ssn_wD_times),
        "ssn_time_avg": np.mean(ssn_times) if ssn_times.size > 0 else 0.0,

        "prox_time_total": np.sum(prox_times),
        "prox_time_avg": np.mean(prox_times) if prox_times.size > 0 else 0.0,

        "lsearch_time_total": np.sum(lsearch_times),
        "lsearch_time_avg": np.mean(lsearch_times) if lsearch_times.size > 0 else 0.0,

        "lsearch_iters_total": total_lsearch_iters,
        "lsearch_iters_avg": total_lsearch_iters / total_ssn_iters if total_ssn_iters > 0 else 0.0,

        "prox_allocs_total": np.sum(prox_allocs),
        "prox_allocs_avg": np.mean(prox_allocs),

        "lsearch_allocs_total": np.sum(lsearch_allocs),
        "lsearch_allocs_avg": np.mean(lsearch_allocs),
    }

    return pd.DataFrame([summary])

def read_all_almlog_csvs(prefix, N, export_path=None):
    
    """
    Reads multiple ALM log CSV files into a single concatenated DataFrame.

    Parameters:
    - prefix: common filename prefix for all CSV files (e.g. 'almlog')
    - N: number of datasets (assumes files are named like 'prefix_dataset1.csv', ..., 'prefix_datasetN.csv')
    - export_path: optional file path to export the combined DataFrame as CSV

    Returns:
    - full_df: a pandas DataFrame containing all rows from the N log files
    """
    
    dfs = []

    for i in range(1, N+1):
        filename = f"{prefix}_dataset{i}.csv"
        df = pd.read_csv(filename)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    if export_path is not None:
        full_df.to_csv(export_path, index=False)

    return full_df




def run_all_experiments(problem_instances, sigma0, tau0, alpha0, AP, SP, LS, output_dir="results"):
    """
    Runs ALM on a list of ProblemInstances, saves weights and a combined summary.

    Parameters:
    - problem_instances: list of ProblemInstance objects
    - sigma0, tau0, alpha0: ALM initialization parameters
    - AP, SP, LS: ALM/SSN/LineSearch parameter objects
    - output_dir: folder to store results (default: 'results')

    Returns:
    - df_all: combined summary DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    summaries = []

    for i, PI in enumerate(problem_instances):
        name = f"dataset{i+1}"
        print(f"🚀 Solving ALM for {name} ...")

        # Run ALM
        almvar, almlog = run_alm(sigma0, tau0, alpha0, PI, AP, SP, LS)

        # Save weights to results folder
        weight_path = os.path.join(output_dir, f"{name}_w.csv")
        np.savetxt(weight_path, almvar.w, delimiter=",")

        # Build and collect summary
        df = summarize_almlog(almlog, name)
        summaries.append(df)

        # Optional: angle diagnostics
        if hasattr(PI, 'w_svm'):
            angle_svm = degrees(acos(dot(PI.w_svm / norm(PI.w_svm), almvar.w / norm(almvar.w))))
            print(f"    ↳ Angle between w_svm and w*: {angle_svm:.2f}°")

        if hasattr(PI, 'w0_shifted'):
            angle_shifted = degrees(acos(dot(PI.w0_shifted / norm(PI.w0_shifted), almvar.w / norm(almvar.w))))
            print(f"    ↳ Angle between w0_shifted and w*: {angle_shifted:.2f}°")

    # Combine and save all summary rows
    df_all = pd.concat(summaries, ignore_index=True)
    df_all_path = os.path.join(output_dir, "summary_all.csv")
    df_all.to_csv(df_all_path, index=False)
    print(f"\n✅ Saved combined summary to: {df_all_path}")

    return df_all

def summarize_sample_efficiency(PI, n_epochs, n_batches, n_pos, n_neg, dataset_name=""):
    total_train_samples = PI.X.shape[1]
    samples_per_batch = n_pos + n_neg
    total_samples_used = samples_per_batch * n_batches * n_epochs

    print("\n--- Sample Efficiency Summary ---")
    print(f"Dataset: {dataset_name}")
    print(f"Total training samples available: {total_train_samples}")
    print(f"Total samples used (SGD batches): {samples_per_batch} × {n_batches} × {n_epochs} = {total_samples_used}")