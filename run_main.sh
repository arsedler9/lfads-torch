#! /bin/bash
echo "--Moving data and config files into temporary directory--"
source activate neurocaas
neurocaas-contrib workflow get-data
neurocaas-contrib workflow get-config

echo "--Parsing paths--"
datapath=$(neurocaas-contrib workflow get-datapath)
configpath=$(neurocaas-contrib workflow get-configpath)
resultpath=$(neurocaas-contrib workflow get-resultpath-tmp)

echo "--Running LFADS--"
source activate lfads-torch
# python /home/ubuntu/lfads-torch/scripts/run_single.py $datapath $configpath $resultpath
python /home/ubuntu/lfads-torch/scripts/run_pbt.py $datapath $configpath $resultpath
source deactivate

echo "--Writing results--"
# neurocaas-contrib workflow put-result -r $resultpath/lfads_output_sess0.h5
neurocaas-contrib workflow put-result -r $resultpath/best_model

source deactivate
