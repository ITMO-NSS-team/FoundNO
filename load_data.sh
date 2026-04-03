#!/bin/bash

current_dir=$(pwd)

if [ "$1" == "flows" ]; then
    huggingface-cli download MikeMasliaev/MultiFlow_poseidon --repo-type dataset --local-dir experiments/data/
elif [ "$1" == "ns_sines" ]; then
    huggingface-cli download MikeMasliaev/Navier-Stokes-Pretraining --repo-type dataset --local-dir experiments/data/
elif [ "$1" == "adv-diff" ]; then
    echo "ADV-DIFF hasn't been implemented in loader yet"
elif [ "$1" == "total" ]; then
    echo "complete multiphys hasn't been implemented in loader yet"
else
    echo "err!"
fi
