#!/bin/bash
#SBATCH -J 2080_GPU
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --mail-type=all
#SBATCH --mem-per-cpu=10240
#SBATCH --mail-user=2513315137@qq.com
sleep 999999