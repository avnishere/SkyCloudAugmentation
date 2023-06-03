# SkyCloudAugmentation
Generative Augmentation using DC-GAN and pix2pix for Sky Cloud Image Segmentation


#### This repository is the codebase for the paper GENERATIVE AUGMENTATION FOR SKY/CLOUD IMAGE SEGMENTATION and will be updated soon. 

This work is in collaboration with [jain15mayank](https://github.com/jain15mayank).


## Create Virtual Environment

```
cd SkyCloudAugmentation

conda create -n <name> python=3.10
conda activate <name>

pip install -r requirements.txt
```

## Training

### DeeplabV3

`cd deeplabv3`

Place/replace values in the hydra `config.yaml` file accordingly. For a quick run: place your train and val dataset folders under `data.train` and `data.val` fields. 

Then, `python train.py` 

you could also override config parameters from the command line. For example: `python train.py optimizer.lr=0.00001 train_params.batch_size=16`


## Testing

### DeeplabV3

`cd deeplabv3`

Place/Replace values under `evaluate` section in `config.yaml`.

Then, `python evaluate.py`

you could also override config parameters from the command line, For example: `python evaluate.py evaluate.model_path='model.pth'`


