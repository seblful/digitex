## 1. Create synthetic data:
Use synthtiger with `numpy==1.26.4` and `Pillow==9.5.0`:

```cmd
python tools/extract_font_charset.py -w 4 resources/font/
```

```cmd
python tools/create_colormap.py --max_k 3 -w 1 resources/image/ resources/colormap/train_colormap.txt
```

```cmd
python tools/create_colormap.py --max_k 3 -w 4 resources/image/ resources/colormap/finetune_colormap.txt
```

```cmd
synthtiger -o results -c 300000 -w 8 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config_train.yaml
```

```cmd
synthtiger -o results -c 10000 -w 8 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config_finetune.yaml
```

create dataset:

## 2. Create dataset
### From synhtiger data

```cmd
uv run src/digitex/training/svtr2/create_dataset.py --source synth --dataset_type lmdb --train_split 0.9 --max_text_length 25 --num_check_images 100
```

### From Label Studio data

```cmd
uv run src/digitex/training/svtr2/create_dataset.py --source ls --dataset_type lmdb --use_aug --train_split 0.8 --max_text_length 25 --num_check_images 100
```

## 2. Train:
### Local:

```cmd
cd C:\Users\seblful\OpenOCR
```

```cmd
.venv\Scripts\activate
```

```cmd
uv run ./tools/train_rec.py --c ./configs/rec/svtrv2/config.yml --o Global.epoch_num=100 Train.sampler.first_bs=128 Train.loader.batch_size_per_card=128 Eval.sampler.first_bs=128 Eval.loader.batch_size_per_card=128
```

### Kaggle GPU P100:

```cmd
!python -m torch.distributed.launch --nproc_per_node=1 tools/train_rec.py --c ./configs/rec/svtrv2/config.yml --o Global.epoch_num=20 Global.pretrained_model=/kaggle/input/svtrv2-tiny/pytorch/default/1/best.pth Optimizer.lr=0.00015 Train.dataset.data_dir_list="[/kaggle/input/svtr2-word-recognition/train]" Eval.dataset.data_dir_list="[/kaggle/input/svtr2-word-recognition/val]" Train.sampler.first_bs=512 Train.loader.batch_size_per_card=512 Eval.sampler.first_bs=512 Eval.loader.batch_size_per_card=512
```

## 3. Finetune:

```cmd
cd C:\Users\seblful\OpenOCR

```

```cmd
.venv\Scripts\activate
```

```cmd
uv run ./tools/train_rec.py --c ./configs/rec/svtrv2/config.yml --o Global.epoch_num=100 Global.pretrained_model=./base-models/svtr2_tiny.pth Train.sampler.first_bs=128 Train.loader.batch_size_per_card=128 Eval.sampler.first_bs=128 Eval.loader.batch_size_per_card=128
```
