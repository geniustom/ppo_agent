import os, glob
import smtplib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt

import smtplib
import time
import re
from os.path import basename


def save_hyperparameters(filenames: list, path_to_file: str,  experiment_name:str=None, save_prob=None, load_prob=None, breadcrumb="# BREADCRUMBS") -> None:
    """
    Saves the lines in between breadcrumbs in all the given filenames. This is used for saving hyperparameters for RL training.
    Parameters
    ----------
    breadcrumb: Writes the lines between {breadcrumb}_START and {breadcrumb}_END.
    """
    with open(path_to_file, "a") as dest:
        if save_prob:
            dest.write(f"Save prob:{save_prob} \n")
        if load_prob:
            dest.write(f"Load prob:{load_prob} \n")
        for filename in filenames:
            with open(filename, "r") as source:
                saving = False
                for line in source:
                    if line.strip() == f"{breadcrumb}_START":
                        dest.write("\n")
                        saving = True
                        continue
                    if line.strip() == f"{breadcrumb}_END":
                        saving = False
                        continue
                    if saving:
                        dest.write(line)
            print(f"{filename} hyperparameters have been saved!")
        print("Information saving is complete!")


def combine_data(treatments):
    treatment_files = dict()
    treatment_combined_data = dict()
    for treatment in treatments:
        exp_runs = glob.glob(f'{treatment}/**/data.txt', recursive=True)
        treatment_files[treatment] = exp_runs

    for key in treatment_files:
        all_run_names = treatment_files[key]
        all_runs = [genfromtxt(run_name, delimiter=',') for run_name in all_run_names]
        max_len = max([len(run) for run in all_runs])
        all_runs_padded = []
        for run in all_runs:
            run = pd.DataFrame(run)
            run = run.df.rolling_mean(10)

            pad = np.empty((max_len - len(run), 2,))
            pad[:] = np.nan
            padded = np.concatenate((run,pad))
            all_runs_padded.append(padded)
        all_runs_padded = np.array(all_runs_padded)

        means = np.nanmean(all_runs_padded, axis=0)
        stds = np.nanstd(all_runs_padded, axis=0)
        step_size = means[:,1][0] 
        steps = [step_size * (i+1) for i in range(max_len)]
        all_results = np.array([means[:,0], stds[:,0], steps])
        treatment_combined_data[key] = all_results
    return treatment_combined_data




if __name__=="__main__":

    dir_name = os.path.split(os.getcwd())[1]
    treatments = [name for name in os.listdir(os.getcwd()) if os.path.isdir(name)]
    
    combined_data = combine_data(treatments)
    fig, ax = plt.subplots()
    ax.set(xlabel='Step Count', ylabel='Reward', title=f'Combined results')

    for treatment_name, data in combined_data.items():
        mean_rewards, mean_std, steps = data[0], data[1], data[2] 
        ax.plot(steps, mean_rewards)
        ax.fill_between(steps, mean_rewards+mean_std, mean_rewards-mean_std,  alpha=0.25)

    plt.show()
    plt.savefig("Difference.png")