#!/bin/bash -x

#SBATCH --time=144:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,vmem:20g

PWD=`pwd`
echo $PWD
activate () {
    . $PWD/ve/bin/activate
}

set_env_vars () {
  PYSTOW_HOME=$PWD/.data
  export PYSTOW_HOME
}

activate
set_env_vars
python src/main.py --dataset "PrimeKG" --models "TransE" --hpo
