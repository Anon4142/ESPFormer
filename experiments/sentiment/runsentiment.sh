#!/bin/bash
#SBATCH --job-name=sentiment
#SBATCH --time=1-00:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-1
#SBATCH --output=outputs/sentiment_seed%a.out
#SBATCH --error=outputs/sentiment_seed%a.err


# Possible seeds
SEEDS=(42 123 545)

# Load Conda if needed
# module load anaconda
# source /path/to/conda.sh
source /opt/apps/rhel9/Anaconda3-2024.02/etc/profile.d/conda.sh

conda activate graphgps

#Set max_seq_len to 512 for IMDB, 128 for TweetEval

# Use SLURM_ARRAY_TASK_ID to index the array of seeds
#Attention type options: esp, vanilla, dif, sink
python one_expe.py --dataset tweet_eval --vocab_file wiki.vocab --max_seq_len 128 --attention_type vanilla --tokenizer sentencepiece --pretrained_model wiki.model --seed "${SEEDS[$SLURM_ARRAY_TASK_ID-1]}"


