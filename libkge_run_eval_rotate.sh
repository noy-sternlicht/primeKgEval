#!/bin/bash -x

#SBATCH --time=70:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-3-40


PWD=`pwd`
echo $PWD
activate () {
    . $PWD/myenv2/bin/activate
}


activate
NOW=$(date +"%F_%H-%M-%S")
LOGFILE="artifacts/log-libkge-$NOW.log"
libKGE/kge/kge/cli.py start libkge_configs/libkge_train_config_rotate.yaml | tee "$LOGFILE"