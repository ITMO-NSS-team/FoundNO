#!/bin/bash

current_dir=$(pwd)

# Check if CUDA is installed on the machine and is availible via path:
if command -v nvcc &> /dev/null
then
    echo "nvcc is found. Checking CUDA Toolkit version..."
    # Execute nvcc --version and store the output
    cuda_version_info=$(nvcc --version)
    
    echo "Full CUDA Toolkit version information:"
    echo "$cuda_version_info"
    
    # Optionally, extract only the version number using grep and awk
    cuda_version=$(echo "$cuda_version_info" | grep "release" | awk '{print $NF}')
    # echo "Extracted CUDA Toolkit version: $cuda_version"
    cuda_prime_version="${cuda_version:1:5}"
    cuda_prime_version="${cuda_prime_version//.}"
    echo "Version key is: $cuda_prime_version"
else
    echo "nvcc command not found. CUDA Toolkit may not be installed or configured in PATH."
    echo "Please ensure CUDA Toolkit is installed and its bin directory is added to your PATH environment variable."
fi

# If unsupported CUDA version, e.g. 13..., exit with an error!
if [[ $cuda_prime_version == 13* ]]; then # 
    echo "Error: Unsupported CUDA version: expected 11.8, 12.1 or 12.8, instead got $cuda_prime_version" >&2
    exit 1
fi

# ADD VERSION LOGIC TO CHECK IF CUDA & TORCH ARE COMPATIBLE (even though there are similar checks in various installations later on)

pip install pandas
pip install packaging
pip install torch==2.4.0+cu$cuda_prime_version torchvision==0.19.0+cu$cuda_prime_version torchaudio==2.4.0+cu$cuda_prime_version --index-url https://download.pytorch.org/whl/cu$cuda_prime_version
pip install neuraloperator==2.0.0
pip install zencfg

apt-get install build-essential
if [ ! -d "causal-conv1d" ]; then
  echo "causal-conv1d does not exist, cloning and installing it."
  git clone https://github.com/Dao-AILab/causal-conv1d.git
fi
cd causal-conv1d
echo "Installing causal_conv1d in $current_dir/causal-conv1d/"
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip3 install -e "$current_dir/causal-conv1d/" --no-build-isolation
cd ..

#if [ ! -d "mamba" ]; then
  #git clone https://github.com/state-spaces/mamba.git
  #cd mamba/
  #git reset --hard d7b1ceb
  #git pull
  #cd ..
#fi

#cd mamba
#echo "Installing mamba in $current_dir/mamba/"
#pip3 install -e "$current_dir/mamba/" --no-build-isolation # --no-deps # "$current_dir/mamba/"
#cd ..

# pip install torch==2.4.0+cu$cuda_prime_version torchvision==0.19.0+cu$cuda_prime_version torchaudio==2.4.0+cu$cuda_prime_version --index-url https://download.pytorch.org/whl/cu$cuda_prime_version
