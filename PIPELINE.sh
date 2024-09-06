#!/usr/bin/env bash

#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=30GB
#SBATCH --time=20:00:00
#SBATCH --job-name=codex
#SBATCH --mail-type=begin,fail,end
#SBATCH --output=./output/output_codex_%j.o
#SBATCH --error=./error/error_codex_%j.e





###### INITIALIZE CONDA ENV CODEX BEFORE RUNNING THIS SCRIPT
# conda activate codex

###### ADJUST THE WORKING DIRECTORY TO THE FOLDER WHERE ALL THE DATA FILES ARE IN #########
WD=./CODES-pipeline/data

###### ALSO ADJUST THE SBATCH REQUIREMENTS ABOVE 

#set the path to the snakefile (only needs to be done in initial setup)
SNAKEFILE=./CODES-pipeline/Snakefile


# set path to the configuration file
CONFIG=$WD/config.yml
# switch to pipeline directory
cd $WD

snakemake -s $SNAKEFILE -d $WD --configfile $CONFIG -c10 --latency-wait 120
