#!/bin/sh
#
# ======= SLURM OPTIONS ======= (user input required)
#
### Specify queue to run
#SBATCH --partition=GPU
### Set the job name
#SBATCH --job-name=path_lstm
### Specify the # of cpus for your job.
#SBATCH --gres=gpu:1
#SBATCH --mem=24gb
### Adjust walltime below (default walltime = 7 days, or 168 hours)
### if you require > 7 days, INCREASE to estimated # hours needed
### if you DON'T require 7 days DECREASE to estimated # hours needed
### (hint: jobs with smaller walltime value tend to run sooner)
#SBATCH --time=24:00:00
### pass the full environment
####### UNABLE TO CONVERT (please review): #PBS -V
# send PBS output to /dev/null  (we redirect it below)
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# ===== END SLURM OPTIONS =====

# ======= APP OPTIONS ======= (user input required)
#
### (REQUIRED) define inputfile here 
INPUTFILE="ConvLSTM.py"

### specify additional Python options
OPTS=""

# ===== END APP OPTIONS =====

module load tensorflow/2.2-anaconda3-cuda10.2
### run job
#module load pymod
cd $SLURM_SUBMIT_DIR
python $OPTS < $INPUTFILE
