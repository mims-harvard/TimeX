import pathlib
import os
import time
import sys
import csv

import numpy as np
import pandas as pd
import torch

from datetime import datetime

from FIT.evaluation.baselines import run_baseline

if __name__ == '__main__':
    explainers = ["fit","ifit"]
    windows = [1,8]
    delays = [0,2]
    cvs = [0,1,2,3,4] #list(range(5))
    #samples = [1, 4, 8]
    log_file = 'paper_experiments.csv'
    #data = "simulation_spike"
    #data = "simulation" #(state)
    data = "simulation_spike" #(switch-feature)

    all_trained = False 
    if not all_trained:
        for delay in delays:
            for cv in cvs:
                
                print("Training Model...")
                # These are all different datasets (requiring retraining of generators and model)
                # Train generator and model
                args = {
                    "data" : data,
                    "train_gen" : False,
                    "train" : True,
                    "skip_explanation" : True,
                    "delay" : delay,
                    "cv" : cv
                }

                start = time.process_time()

                run_baseline(**args)

                end = time.process_time()
                time_elapsed = end - start
                print(f"Train Model: {time_elapsed:.2f} seconds")

                print("Training JointFeatureGenerator")
                args = {
                    "data" : data,
                    "explainer_name" : "fit",
                    "train_gen" : True,
                    "train" : False,
                    "N" : 1,
                    "delay" : delay,
                    "cv" : cv
                }

                print(args)

                start = time.process_time()

                run_baseline(**args)

                end = time.process_time()
                time_elapsed = end - start
                print(f"JointFeatureGenerator: {time_elapsed:.2f} seconds")
                
                for window in windows:
                    print("Training FeatureGenerator")
                    args = {
                        "data" : data,
                        "explainer_name" : "ifit",
                        "train_gen" : True,
                        "train" : False,
                        "N" : window,
                        "delay" : delay,
                        "cv" : cv
                    }

                    print(args)

                    start = time.process_time()

                    run_baseline(**args)

                    end = time.process_time()
                    time_elapsed = end - start
                    print(f"FeatureGenerator: {time_elapsed:.2f} seconds")

    #sys.exit()
    
    
    print("=====")
    print("Evaluating Explanations")
    print("=====")
    
    header = ['data', 'target_delay', 'cv', 'explainer', 'window', 'samples', 'auc', 'auprc', 'runtime', 'experiment_time']
    with open(log_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    for delay in delays:
        for cv in cvs:        
            for explainer_name in explainers:
                for window in windows: 
                    #for n_samples in samples:
                    print("Running Explanations...")

                    args = {
                        "data" : data,
                        "explainer_name" : explainer_name,
                        "train_gen" : False,
                        "train" : False,
                        "N" : window,
                        "delay" : delay,
                        "samples" : 3 if explainer_name == "ifit" else 10,
                        #"samples" : n_samples,
                        "cv" : cv
                    }

                    print(args)

                    start = time.process_time()

                    auc, auprc = run_baseline(**args)

                    end = time.process_time()
                    time_elapsed = end - start

                    print(f"Explanation: [{auc:.3f}, {auprc:.3f}, {time_elapsed:.2f}]")
                    row = [args['data'], args['delay'], args['cv'], args['explainer_name'], args['N'], args["samples"], auc, auprc, time_elapsed, str(datetime.now())]

                    # Write out results to log
                    with open(log_file, 'a', encoding='UTF8') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    


