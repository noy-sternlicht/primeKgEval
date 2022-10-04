#!/bin/bash -x

#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-1-10

PWD=`pwd`
echo $PWD
activate () {
    . $PWD/myenv/bin/activate
}

set_env_vars () {
  PYSTOW_HOME=$PWD/.data
  export PYSTOW_HOME
}

activate
set_env_vars
python main.py --dataset "Nations" --models "TransE"
