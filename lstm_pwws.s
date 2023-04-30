#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=lstm_pwws
#SBATCH --mail-type=END
#SBATCH --mail-user=ql2078@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --dependency=afterok:32924426

module purge
module load anaconda3/2020.07

RUNDIR=$SCRATCH/DefenseEval/lstm_pwws/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

cd $RUNDIR
source activate ../../../defeval
python ../../adv_train_1.py
