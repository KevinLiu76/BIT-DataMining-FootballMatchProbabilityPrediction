运行以下命令训练模型：

```
python train.py \
    --train_dataset_path ../data/X_train.csv \
    --val_dataset_path ../data/X_val.csv \
    --save_model_dir ../data/dumps \
    --epochs 100 \
    --batch_size 512
```

