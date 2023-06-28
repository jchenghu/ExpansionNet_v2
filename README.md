## ExpansionNet v2:  Block Static Expansion in fast end to end training for Image Captioning

Implementation code for [ExpansionNet v2: Block Static Expansion
in fast end to end training for Image Captioning](https://arxiv.org/abs/2208.06551v3). <br>


## Demo

You can test the model generic images (not included in COCO) downloading
 the checkpoint [here](https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3?usp=sharing)
and launching the script `demo.py`:
``` 
python demo.py \
     	--load_path your_download_folder/rf_model.pth \
     	--image_paths your_image_path/image_1 your_image_path/image_2 ...

```
Some examples:

<p align="center">
  <img src="./demo_results.png" width="550"/>
</p>

images are available in `demo_material`.


## ONNX & TensorRT

The model supports now ONNX conversion and deployment with TensorRT. 
The graph can be generated using `onnx4tensorrt/convert2onnx.py`.
Its execution mainly requires the `onnx` package but the `onnx_runtime` and `onnx_tensorrt` packages are
optionally used for testing purposes (see `convert2onnx.py` arguments).

Assuming Generic conversion commands:
```
python onnx4tensorrt/convert2onnx.py --onnx_simplify true --load_model_path <your_path> &> output_onnx.txt &
python onnx4tensorrt/onnx2tensorrt.py &> output_tensorrt.txt &
```
the engine will be found as `model_engine.trt`.
Currently working only in FP32.

## Results

Results in the online evaluation server:
 <table>
  <tr>
    <th>Captions</th>
    <th>B1</th>
    <th>B2</th>
    <th>B3</th>
    <th>B4</th>
    <th>Meteor</th>
    <th>Rouge-L</th>
    <th>CIDEr-D</th>
  </tr>
  <tr>
    <th>c40</th>
    <td>96.9</td>
    <td>92.6</td>
    <td>85.0</td>
    <td>75.3</td>
    <td>40.1</td>
    <td>76.4</td>
    <td>140.8</td>
  </tr>
  <tr>
    <th>c5</th>
    <td>83.3</td>
    <td>68.8</td>
    <td>54.4</td>
    <td>42.1</td>
    <td>30.4</td>
    <td>60.8</td>
    <td>138.5</td>
  </tr>
</table>

Results on the Karpathy test split:
 <table>
  <tr>
    <th>Model</th>
    <th>B@1</th>
    <th>B@4</th>
    <th>Meteor</th>
    <th>Rouge-L</th>
    <th>CIDEr-D</th>
    <th>Spice</th>
  </tr>
  <tr>
    <td>Ensemble</td>
    <td>83.5</td>
    <td>42.7</td>
    <td>30.6</td>
    <td>61.1</td>
    <td>143.7</td>
    <td>24.7</td>
  </tr>
  <tr>
    <td>Single</td>
    <td>82.8</td>
    <td>41.5</td>
    <td>30.3</td>
    <td>60.5</td>
    <td>140.4</td>
    <td>24.5</td>
  </tr>
</table>

Predictions examples:

<p align="center">
  <img src="./results_image.png" width="700"/>
</p>



## Training

In this guide we cover all the training steps reported in the paper and
provide the commands to reproduce our work.

#### Requirements

* python >= 3.7
* numpy
* Java 1.8.0
* pytorch 1.9.0
* h5py

#### Data preparation

MS-COCO 2014 images can be downloaded [here](https://cocodataset.org/#download), 
the respective captions are uploaded in our online [drive](https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3?usp=sharing)
and the backbone can be found [here](https://github.com/microsoft/Swin-Transformer). All files, in particular
the 3 json files and the backbone are suggested to be moved in `github_ignore_materal/raw_data/` since commands provided
in the following steps assume these files are placed in that directory.


#### Premises

For the sake of transparency (at the cost of possibly being overly verbose)
the complete commands are shown below, but only few arguments deserve a little
bit of care for the reproduction of our work while most of them are automatically handled.

Logs are stored in `output_file.txt`, which is continuously updated
until the process is complete (in Linux it may be handy the command `watch -n 1 tail -n 30 output_file.txt`). It is overwritten in each training phase, thus,
before moving to the next one, make sure to save or make a copy if needed.

Lastly, in some configurations the batch size may look different compared to the one
reported in the paper when argument `num_accum` is specified (default is 1). This is only
a visual subtlety, which means that gradient accumulation is performed in order to satisfy 
the memory constraints of 40GB RAM of a single GPU.

#### 1. Cross Entropy Training: Features generation

First we generate the features for the first training step:
```
cd ExpansionNet_v2_src
python data_generator.py \
    --save_model_path ./github_ignore_material/raw_data/swin_large_patch4_window12_384_22k.pth \
    --output_path ./github_ignore_material/raw_data/features.hdf5 \
    --images_path ./github_ignore_material/raw_data/MS_COCO_2014/ \
    --captions_path ./github_ignore_material/raw_data/ &> output_file.txt &
```
Even if it's suggested not to do so, the `output_path` argument can be replaced with the desired destination (this would require
changing the argument `features_path` in the next commands as well). Since it's a pretty big
file (102GB), once the first training is completed, it will be automatically overwritten by
the remaining operations in case the default name is unchanged.


#### 2. Cross-Entropy Training: Partial Training

In this step the model is trained using the Cross Entropy loss and the features generated
in the previous step:
```
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --seed 775533 --optim_type radam --sched_type custom_warmup_anneal  \
    --warmup 10000 --lr 2e-4 --anneal_coeff 0.8 --anneal_every_epoch 2 --enc_drop 0.3 \
    --dec_drop 0.3 --enc_input_drop 0.3 --dec_input_drop 0.3 --drop_other 0.3  \
    --batch_size 48 --num_accum 1 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end False --features_path ./github_ignore_material/raw_data/features.hdf5 --partial_load False \
    --print_every_iter 11807 --eval_every_iter 999999 \
    --reinforce False --num_epochs 8 &> output_file.txt &
```

#### 3. Cross-Entropy Training: End to End Training

The following command trains the entire network in the end to end mode. However,
one argument need to be changed according to the previous result, the
checkpoint name file. Weights are stored in the directory `github_ignore_materal/saves/`,
with the prefix `checkpoint_ ... _xe.pth` we will refer it as `phase2_checkpoint` below and in
the later step:
```
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533   --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 3e-5 --anneal_coeff 0.55 --anneal_every_epoch 1 --enc_drop 0.3 \
    --dec_drop 0.3 --enc_input_drop 0.3 --dec_input_drop 0.3 --drop_other 0.3  \
    --batch_size 16 --num_accum 3 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --images_path ./github_ignore_material/raw_data/MS_COCO_2014/ --partial_load True \
    --backbone_save_path ./github_ignore_material/raw_data/swin_large_patch4_window12_384_22k.pth \
    --body_save_path ./github_ignore_material/saves/phase2_checkpoint \
    --print_every_iter 15000 --eval_every_iter 999999 \
    --reinforce False --num_epochs 2 &> output_file.txt &
```
In case you are interested in the network's weights at the end of this stage, 
before moving to the self-critical learning, rename the checkpoint file from `checkpoint_ ... _xe.pth` into something 
else like `phase3_checkpoint` (make sure to change the prefix) otherwise it will 
be overwritten during step 5.

#### 4. CIDEr optimization: Features generation

This step generates the features for the reinforcement step:
```
python data_generator.py \
    --save_model_path ./github_ignore_material/saves/phase3_checkpoint \
    --output_path ./github_ignore_material/raw_data/features.hdf5 \
    --images_path ./github_ignore_material/raw_data/MS_COCO_2014/ \
    --captions_path ./github_ignore_material/raw_data/ &> output_file.txt &
```

#### 5. CIDEr optimization: Partial Training

The following command performs the partial training using the self-critical learning:
```
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 24 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end False --partial_load True \
    --features_path ./github_ignore_material/raw_data/features.hdf5 \
    --body_save_path ./github_ignore_material/saves/phase3_checkpoint.pth \
    --print_every_iter 4000 --eval_every_iter 99999 \
    --reinforce True --num_epochs 9 &> output_file.txt &
```
We refer to the last checkpoint produced in this step as `phase5_checkpoint`,
it should already achieve around 139.5 CIDEr-D on both Validaton and Test set, however
it can be still improved by a little margin with the following optional step.


#### 6. CIDEr optimization: End to End Training

This last step again train the model in an end to end fashion, however it is optional since it only slightly improves the performances:
```
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --anneal_coeff 1.0 --lr 2e-6 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 24 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --images_path ./github_ignore_material/raw_data/MS_COCO_2014/ --partial_load True \
    --backbone_save_path ./github_ignore_material/raw_data/phase3_checkpoint \
    --body_save_path ./github_ignore_material/saves/phase5_checkpoint \
    --print_every_iter 15000 --eval_every_iter 999999 \
    --reinforce True --num_epochs 1 &> output_file.txt &
```

## Evaluation

In this section we provide the evaluation scripts. We refer to the
last checkpoint as `phase6_checkpoint`. In case the previous training 
procedures have been skipped, 
weights of one of the ensemble's model can be found [here](https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3?usp=sharing).
```
python test.py --N_enc 3 --N_dec 3 --model_dim 512 \
    --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True \
    --save_model_path ./github_ignore_material/saves/phase6_checkpoint
```
The option `is_end_to_end` can be toggled according to the model's type.


## Citation

If you find this repository useful, please consider citing (no obligation):
```
@article{hu2022expansionnet,
  title={ExpansionNet v2: Block Static Expansion in fast end to end training for Image Captioning},
  author={Hu, Jia Cheng and Cavicchioli, Roberto and Capotondi, Alessandro},
  journal={arXiv preprint arXiv:2208.06551},
  year={2022}
}
```

## Acknowledgements

We thank the PyTorch team and the following repositories:
* https://github.com/microsoft/Swin-Transformer
* https://github.com/ruotianluo/ImageCaptioning.pytorch
* https://github.com/tylin/coco-caption

special thanks to the work of [Yiyu Wang et al](https://arxiv.org/abs/2203.15350).

We also thank the user [@shahizat](https://github.com/shahizat) for the suggestion of ONNX and TensorRT conversions.

