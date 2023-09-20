# Installation
Install Anaconda or Miiniconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
```bash
git clone https://github.com/automl/SAFS-RL.git
cd /SAFS-RL
conda create --name myenv python=3.8.2
conda activate myenv
pip install -r requirements.txt
```
# To run on cluster
```bash
# Change to my work dir
cd $SLURM_SUBMIT_DIR
module load Miniconda3
module load GCCcore/.9.3.0
module load Python/3.8.2
module load cuDNN/8.2.2.26-CUDA-11.4.1

# Replace user with your username and myenv with your environment name
conda activate /bigwork/user/miniconda3/envs/myenv

# Run RL environment
python safs_rl/environemnt.py
```

# Example of SLURM script
```bash
#!/bin/bash -l
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output test_gpu.out
#SBATCH --error test_gpu.err
 
# Change to my work dir
cd $SLURM_SUBMIT_DIR
module load Miniconda3
module load GCCcore/.9.3.0
module load Python/3.8.2
module load cuDNN/8.2.2.26-CUDA-11.4.1
conda activate /bigwork/nhwppetw/miniconda3/envs/safs-rl

# Run GPU application
python safs_rl/braxenv.py
```

