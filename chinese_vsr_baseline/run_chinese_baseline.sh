
acc_cfg=~/.cache/huggingface/accelerate/default_config.yaml

ngpu=8

cfg_ph=./cfgs/TV101/vsr_decoding_conformer.yaml

# start training from scratch
accelerate launch --config_file ${acc_cfg} ./decoding_train.py ${cfg_ph} -n ${ngpu}  --print_params  --find_unused_parameters

# continue training
logdir=./.checkpoints/TV101/vsr/vsr_decoding_conformer/12-08-00-14-32
cfg_ph=${logdir}/config.yaml
accelerate launch --config_file ${acc_cfg} ./decoding_train.py ${cfg_ph} -n ${ngpu}  --print_params  --find_unused_parameters --log_dir ${logdir} --last_epoch 3

