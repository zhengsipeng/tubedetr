CUBLAS_WORKSPACE_CONFIG=:16:8 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --ema \
    --load=pretrained_resnet101_checkpoint.pth \
    --combine_datasets "ego4d" --combine_datasets_val "ego4d" \
    --dataset_config config/ego4d.json \
    --output-dir "saved" \
    --resolution 384 \
    --num_workers 0 