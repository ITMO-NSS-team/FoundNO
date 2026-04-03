#!/bin/bash

current_dir=$(pwd)
echo "The current directory is: $current_dir"

# Description & an example of inputs: 

# Path to csv with non-zero sources: $1 = preset/sample_0000_nonzero_points.csv;
# Path to hdf5-file with initial conditions (separated or with further model predictions): $2 = dataset_on_off_2/dataset_val_anizotrop.hdf5;
# Path to model: $3 = mamba_gpn_120.pt;
# Path to directory, where the output will be stored: $4 = preset/;
# Sample number, guiding the selection of sample from the hdf5-file: $5 = "0";
# Prediction horizon: $6 = "37".

python3 "$current_dir/mamba_fno_inference.py" --csv_path "$current_dir/$1" --initial_conditions_path "$current_dir/$2" --model_weights $"${current_dir}/$3" --output_dir "$current_dir/$4" --sample_idx "$5" --time_horizon "$6"
