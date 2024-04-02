

# Class Similarity Transition: Decoupling Class Similarities and Imbalance from Generalized Few-shot Segmentation

This repository contains the code for our paper, [Class Similarity Transition: Decoupling Class Similarities and Imbalance from Generalized Few-shot Segmentation]().

> **Abstract:** *In Generalized Few-shot Segmentation (GFSS), a model is trained with a large corpus of base class samples and then adapted on limited samples of novel classes. This paper focuses on the relevance between base and novel classes, and improves GFSS in two aspects: 1) mining the similarity between base and novel classes to promote the learning of novel classes, and 2) mitigating the class imbalance issue caused by the volume difference between the support set and the training set. Specifically, we first propose a similarity transition matrix to guide the learning of novel classes with base class knowledge. Then, we leverage the Label-Distribution-Aware Margin (LDAM) loss and Transductive Inference to the GFSS task to address the problem of class imbalance as well as overfitting the support set. In addition, by extending the probability transition matrix, the proposed method can mitigate the catastrophic forgetting of base classes when learning novel classes. With a simple training phase, our proposed method can be applied to any segmentation network trained on base classes. We validated our methods on the adapted version of OpenEarthMap. Compared to existing GFSS baselines, our method excels them all from 3\% to 7\% and ranks second in the OpenEarthMap Land Cover Mapping Few-Shot Challenge at the completion of this paper.*

## &#x1F3AC; Getting Started

### :one: Requirements
We used `Python 3.9` in our experiments and the list of packages is available in the `requirements.txt` file. You can install them using `pip install -r requirements.txt`.

### :two: Download data

#### Pre-processed data from drive

We use a [adapted version](https://zenodo.org/records/10828417) of OpenEarthMap datasets. You can download the full .zip and directly extract it in the `data/` folder.

#### From scratch

Alternatively, you can prepare the datasets yourself. Here is the structure of the data folder for you to reproduce:

```
data
├── trainset
│   ├── images
│   └── labels
│   
├── valset
|   ├── images
|   └── labels
|
├── testset
|   ├── images
|   └── labels
|
├── train.txt
├── stage1_val.txt
├── test.json
└── val.json

```

<!-- ### :three: Download pre-trained models

#### Pre-trained backbone and models
We provide the pre-trained backbone and models at - https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/tree/main. You can download them and directly extract them at the root of `pretrain/`. -->

## &#x1F5FA; Overview of the repo

Default configuration files can be found in `config/`. Data are located in `data/` contains the train/val dataset. All the codes are provided in `src/`. Testing script is located at the root of the repo.

## &#x2699; Training 

We use [ClassTrans-Train](https://github.com/earth-insights/ClassTrans-Train) to train models on base classes. We suggest to skip this step and directly use this **[checkpoint](https://drive.google.com/file/d/1H9Z9bLU46tDoqXHEhc4BduQ_Vs2RqGvM/view?usp=sharing)** to reimplement our results.

## &#x1F9EA; Testing

```bash
# Creating a soft link from `ClassTrans-Train/segmentation_models_pytorch` to `ClassTrans/segmentation_models_pytorch`
ln -s /your/path/ClassTrans-Train/segmentation_models_pytorch /your/path/ClassTrans
# Creating a soft link from `ClassTrans-Train/weight` to `ClassTrans/weight`
ln -s /your/path/ClassTrans-Train/weight /your/path/ClassTrans
# Run the testing script
bash test.sh
```

## &#x1F9CA; Post-processing

In `test.py`, you can find some post-processing of the prediction masks with extra input files, which are obtained via a vision-language model [APE](https://arxiv.org/abs/2312.02153) and a class-agnostic mask refinement model [CascadePSP](https://arxiv.org/abs/2005.02551). We provide these files in the `Class-Trans/post-process` directory. If you want to reproduce our results step by step, you can refer to the following:

### APE

APE is a vision-language model which can conduct open-vocabulary detection and segmentation. We directly use the released checkpoint [APE-D](https://huggingface.co/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth) to infer the base class `sea, lake, & pond` and the novel classes `vehicle & cargo-trailer` and `sports field`, using the following commands:

```bash
# sea, lake, & pond
python demo/demo_lazy.py --config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py --input data/cvpr2024_oem_ori_png/*.png --output output/cvpr2024_oem_ori_thres-0.12_water/ --confidence-threshold 0.12 --text-prompt 'water' --with-sseg --opts train.init_checkpoint=model_final.pth model.model_vision.select_box_nums_for_evaluation=500 model.model_vision.text_feature_bank_reset=True

# vehicle & cargo-trailer
python demo/demo_lazy.py --config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py --input data/cvpr2024_oem_crop_256-128/*.png --output output/cvpr2024_oem_crop-256-128_thres-0.1_car/ --confidence-threshold 0.1 --text-prompt 'car' --with-sseg --opts train.init_checkpoint=model_final.pth model.model_vision.select_box_nums_for_evaluation=500 model.model_vision.text_feature_bank_reset=True

# sports field
python demo/demo_lazy.py --config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py --input data/cvpr2024_oem_crop_256-128/*.png --output output/cvpr2024_oem_crop-256-128_thres-0.2_sportfield/ --confidence-threshold 0.2 --text-prompt 'sports field,basketball field,soccer field,tennis field,badminton field' --with-sseg --opts train.init_checkpoint=model_final.pth model.model_vision.select_box_nums_for_evaluation=500 model.model_vision.text_feature_bank_reset=True
```

Before executing the above commands, please make sure that you have successfully built the APE environment and sliced the original image into appropriate image tiles:

1. Please refer [here](https://github.com/shenyunhang/APE) to build APE's reasoning environment, we highly recommend using **docker** to build it.

2. Convert the RGB images from '.tif' format to '.png' format and use `image2patch.py` script to generate image tiles.

After reasoning with APE, use the following commands to compose the results of the image tiles into the whole image:

```bash
# get semantic mask from instance mask
python tools/get_mask_from_instance.py
# get the complete result for the whole image
python tools/patch2image.py
```

Note: We have confirmed that using the foundation model is consistent with the challenge rules.

### Mask Refinement

We use [CascadePSP](https://github.com/hkchengrex/CascadePSP) to refine the mask of building type 1 & 2

```bash
# install segmentation_refinement
pip install segmentation_refinement
# get refined mask of building type 1 & 2
python tools/mask_refinement.py 
```

### &#x1F4CA; Results

| Class             | IoU      |
|-------------------|----------|
| Tree              | 68.94964 |
| Rangeland         | 49.81997 |
| Bareland          | 32.84904 |
| Agric land type 1| 53.61771 |
| Road type 1       | 57.60924 |
| Sea, lake, & pond| 53.97921 |
| Building type 1   | 55.54934 |
|-------------------|----------|
| Vehicle & cargo-trailer| 37.24685 |
| Parking space     | 32.26357 |
| Sports field      | 49.98770 |
| Building type 2   | 52.10971 |
| mIoU for base classes | 53.19631 |
| mIoU for novel classes| 42.90196 |
| Weighted average of mIoU scores for base and novel classes | 47.01970 |

The weighted average is calculated using `0.4:0.6 => base:novel` based on SOA GFSS baseline.



## &#x1F64F; Acknowledgments

We gratefully thank the authors of [BAM](https://github.com/chunbolang/BAM), [DIAM](https://github.com/sinahmr/DIaM), [APE](https://github.com/shenyunhang/APE), [CascadePSP](https://github.com/hkchengrex/CascadePSP) and [PyTorch Semantic Segmentation](https://github.com/hszhao/semseg) from which some parts of our code are inspired.

## &#x1F4DA; Citation

If you find this project useful, please consider citing:

```bibtex

```