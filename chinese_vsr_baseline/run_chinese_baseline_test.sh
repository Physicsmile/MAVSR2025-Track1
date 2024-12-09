acc_cfg=~/.cache/huggingface/accelerate/default_config.yaml

train_path=./.checkpoints/TV101/vsr/vsr_decoding_conformer/12-08-00-14-32

nbest=5
ngpu=8
max_epoch=100

start_epoch=1
end_epoch=100

# inference
cfg_ph=${train_path}/config.yaml
python3 ./decoding_eval.py ${cfg_ph} val --start_epoch ${start_epoch} --end_epoch ${end_epoch} --train_path ${train_path} --nprocs 8 --gpu_idxs 0,1,2,3,4,5,6,7 
# test on mov20
cfg_ph=${train_path}/config_mov20.yaml
python3 ./decoding_mov20.py ${cfg_ph} val --model_average_max_epoch ${max_epoch} --nbest ${nbest} --train_path ${train_path} --nprocs 8 --gpu_idxs 0,1,2,3,4,5,6,7 
