# SegFormer

## 1. Create dataset

```cmd
uv run src/digitex/training/segformer/create_dataset.py --train_split 0.8 --augment --aug_images 100 --visualize --vis_images 50
```

## 2. Train

```cmd
uv run src/digitex/training/segformer/train.py
```
