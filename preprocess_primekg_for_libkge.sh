#!/bin/bash -x

#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4069m


PWD=`pwd`
echo $PWD
activate () {
    . $PWD/myenv2/bin/activate
}


activate
python Data/primekg/split.py
mv Data/primekg/train.txt Data/primekg/valid.txt Data/primekg/test.txt libKGE/kge/data/primekg
python libKGE/kge/data/preprocess/preprocess_default.py libKGE/kge/data/primekg
