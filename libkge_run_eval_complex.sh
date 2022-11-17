#!/bin/bash -x

#SBATCH --time=144:00:00
#SBATCH --ntasks=5
#SBATCH --gres=gpu:2,vmem:10g


PWD=`pwd`
echo $PWD
activate () {
    . $PWD/myenv2/bin/activate
}


activate
NOW=$(date +"%F_%H-%M-%S")
LOGFILE="artifacts/log-libkge-complex-$NOW.log"
libKGE/kge/kge/cli.py start libkge_configs/libkge_train_config_complex.yaml --search.device_pool cuda:0,cuda:1 --search.num_workers 10 | tee "$LOGFILE"
