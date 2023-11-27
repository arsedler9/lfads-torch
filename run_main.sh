#! /bin/bash
echo "--Moving data and config files into temporary directory--"
source activate neurocaas
neurocaas-contrib workflow get-data
neurocaas-contrib workflow get-config

echo "--Parsing paths--"
datapath=$(neurocaas-contrib workflow get-datapath)
configpath=$(neurocaas-contrib workflow get-configpath)
resultpath=$(neurocaas-contrib workflow get-resultpath-tmp)

echo "--Running AutoLFADS--"
source activate lfads-torch
python /home/ubuntu/lfads-torch/scripts/run_pbt.py $datapath $configpath $resultpath
source deactivate

echo "--Writing results--"
cd $resultpath/best_model
zip -r autolfads.zip *
neurocaas-contrib workflow put-result -r autolfads.zip

source deactivate
