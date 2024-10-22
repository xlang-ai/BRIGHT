#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=eval ## CHANGE JOBNAME HERE
#SBATCH --array=0-2

# Remove one # to uncommment
#SBATCH --output=./joblog/%x-%A_%a.out                          ## Stdout
#SBATCH --error=./joblog/%x-%A_%a.err                           ## Stderr

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --constraint=gpu80
# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=ALL
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=nobody

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Array Job ID                   = $SLURM_ARRAY_JOB_ID"
echo "Array Task ID                  = $SLURM_ARRAY_TASK_ID"
echo "Cache                          = $TRANSFORMERS_CACHE"

source env/bin/activate

IDX=$SLURM_ARRAY_TASK_ID
NGPU=$SLURM_GPUS_ON_NODE
if [[ -z $SLURM_ARRAY_TASK_ID ]]; then
    IDX=1
    NGPU=1
fi
PORT=$(shuf -i 30000-65000 -n 1)
echo "Port                          = $PORT"

export OMP_NUM_THREADS=8

MODELS=(
    bge
    qwen
    qwen2
    e5
    grit
    inst-l
    inst-xl
    sbert
    sf
)
MODEL="${MODELS[$IDX / 3]}"

DATASETS=(
    theoremqa_theorems
    theoremqa_questions
    aops
)
DATASET="${DATASETS[$IDX % 3]}"


OPTIONS=""
echo "Options                       = $OPTIONS"

python run.py --model $MODEL --task $DATASET

echo "finished with $?"

wait;

#echo "done, check $OUTPUT_DIR for outputs"

#exit 0

