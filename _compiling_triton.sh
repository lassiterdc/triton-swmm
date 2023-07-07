#!/bin/bash
#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
# SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATC H -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name # can use dev for testing
#SBATCH -t 6:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1	# Array of jobs (e.g., 1-1000)
# SBATCH --mem-per-cpu=36000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1     # see note below
#SBATCH --ntasks-per-node=1   # 

# A note on the number of GPU devices (line 2, can be 1-8) and the number of MPI ranks (line 3).
# It is usually recommended that the two should be identical

# To request more GPUs you�ll need to request more nodes, e.g.:
# #SBATCH -N 2
#  We only have 12 A100 nodes in total right now, and to prevent any single user from dominating the queue we impose a limit of 32 GPUs per job.

# interactive job: ijob -c 1 -A quinnlab_paid -p gpu --time=0-04:00:00 --gres=gpu:a100:1

module purge
module load nvompic
make clean && make hpc_gpu

# to run the model: 
# srun /project/quinnlab/dcl3nd/TRITON/triton/input/cfg/case01.cfg # [path to config file]

