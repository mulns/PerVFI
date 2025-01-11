# Train PerVFI original version
accelerate launch --config_file 'configs/multi-gpu.yaml' \
           train.py -c configs/pvfi-v00.yml 
# Train PerVFI-vb version (without normalizing flow network)
accelerate launch --config_file 'configs/multi-gpu.yaml' \
           train.py -c configs/pvfi-vb.yml 
