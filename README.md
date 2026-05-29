Official implementation of our paper:

Bidomain Multi-order Modeling for Image Dehazing

## Dataset Preparation

Download RESIDE ITS and organize as:

```
./DataFiles/ITS/
└── train_indoor/
    ├── hazy/   # *.png
    └── gt/     # *.png
```

## Training

```bash
python main.py \
    --dir_data ./DataFiles \
    --model sffn \
    --lr 2e-4 \
    --save sffn_its \
    --patch_size 256 \
    --batch_size 8 \
    --loss '1*L1+0.1*Rea+0.1*Ima' \
    --lr_decay 200 \
    --epochs 1000
```

Resume from checkpoint:

```bash
python main.py \
    --model sffn \
    --save sffn_its \
    --load sffn_its \
    --resume -1
```

## Testing

```bash
python evaluate_its.py \
    --model sffn \
    --pre_train ../experiment/sffn_its/model/model_best.pt \
    --dataset indoor \
    --batch_size 4
```
