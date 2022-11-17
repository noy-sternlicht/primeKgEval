#!/bin/bash -x

#SBATCH --time=144:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000


PWD=`pwd`
echo $PWD
activate () {
    . $PWD/myenv2/bin/activate
}


activate
NOW=$(date +"%F_%H-%M-%S")
LOGFILE="artifacts/log-libkge-complex-$NOW.log"
libKGE/kge/kge/cli.py start libkge_configs/libkge_train_config_complex.yaml | tee "$LOGFILE"