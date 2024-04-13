# BP_snuparek
This repository contains the code acompanying my bachelors thesis. It contains the inference script and guide how to train your own models to recreate the results of my testing. 

# Running Inference script
To run the inference script on a DICOM scan create a virtual environment and install the requirements in requirements.txt. You can download the pretrained model weights from here: [Download pretrained weights](https://drive.google.com/file/d/14PvBIE4N53O7fyAwLrcyNabf8i94iqwT/view?usp=sharing)
Or you can train your own model using the description below.

Set paths to DICOM file and model weights at the beginning of the script

''' bash
dicom_path = 'path_to_dicom_file'
model_path = 'path_to_model_weights'
'''

# Training Swin-UNETr
For further details visit original:
[Repository](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21)

A Swin UNETR network with standard hyper-parameters for brain tumor semantic segmentation (BraTS dataset) is be defined as:

``` bash
model = SwinUNETR(img_size=(128,128,128),
                  in_channels=4,
                  out_channels=3,
                  feature_size=48,
                  use_checkpoint=True,
                  )
```


Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch native AMP package:
``` bash
python main.py
--feature_size=48
--batch_size=1
--logdir=unetr_test_dir
--fold=0
--optim_lr=1e-4
--lrschedule=warmup_cosine
--infer_overlap=0.5
--save_checkpoint
--val_every=10
--json_list='./jsons/brats21_folds.json'
--data_dir=/brats2021/
--use_checkpoint
--noamp
```

## Training from scratch on single GPU with gradient check-pointing and without AMP

To train a `Swin UNETR` from scratch on a single GPU with gradient check-pointing and without AMP:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
```

## Finetuning on single GPU with gradient check-pointing and without AMP

To finetune a `Swin UNETR`  model on a single GPU on fold 1 with gradient check-pointing and without amp,
the model path using `pretrained_dir` and model  name using `--pretrained_model_name` need to be provided:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --noamp --pretrained_model_name=<model-name> \
--pretrained_dir=<model-dir> --fold=1 --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
```
