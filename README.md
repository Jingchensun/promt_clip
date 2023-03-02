# Prompt Learning for CLIP Model (CoOp)


### Step 1: Installation
Create a conda environment and install dependencies:
```bash
git clone git@github.com:Jingchensun/promt_clip.git
cd promt_clip

conda create -y -n torch180 python=3.8
conda activate torch180
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

pip install -r requirements.txt

```

### Step 2: Dataset
All datasets are set in A5000 server. You just need to create a soft link:
```bash
cd prompt_tipadapter
ln -s /data/jason/data/coopdata data/
```

Or follow [DATASETS.md](DATASETS.md) to install the datasets from [CoOp](https://github.com/KaiyangZhou/CoOp/tree/main/datasets) for multitask prompt initialization. Or run the following script(11 datasets, include ImageNet): 
```bash
bash scripts/data.sh
```

If you only failed on the ImageNet, you can simply run the following script: 
```bash
bash scripts/imagenet.sh
```







### Step 3-1: Few Shot Training
All you need is `CoOp/scripts/coop/main.sh`, which contains six input arguments.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CoOp/configs/datasets/`.

`CFG` means which config file to use, such as `rn50`, `rn101` or `vit_b32` (see `CoOp/configs/trainers/CoOp/`). Note that for ImageNet, we use `CoOp/configs/trainers/CoOp/*_ep50.yaml` for all settings (please follow the implementation details shown in the paper).

Below we provide examples on how to run CoOp on Caltech101:

```bash
- 1 shot: `bash scripts/coop/main.sh caltech101 vit_b16_ctxv1 end 16 1 False`
- 2 shots: `bash scripts/coop/main.sh caltech101 vit_b16_ctxv1 end 16 2 False`
- 4 shots: `bash scripts/coop/main.sh caltech101 vit_b16_ctxv1 end 16 4 False`
- 8 shots: `bash scripts/coop/main.sh caltech101 vit_b16_ctxv1 end 16 8 False`
```

other datasets (200 epochs):
```bash
bash scripts/coop/main.sh fgvc_aircraft vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh stanford_cars vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh food101 vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh dtd vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh eurosat vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh caltech101 vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh sun397 vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh ucf101 vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh oxford_pets vit_b16_ctxv1 end 16 8 False
bash scripts/coop/main.sh oxford_flowers vit_b16_ctxv1 end 16 8 False
```

ImageNet (50 epochs)
```bash
bash scripts/coop/main.sh imagenet vit_b16_ep50_ctxv1.yaml end 16 8 False
```





### Step 3-2: Parse the test results
After the experiments are finished, you can use `parse_test_res.py` to calculate the average results instead of manually looking into the log files. Say the structure of `output/` is

```
output
|–– caltech101/
|   |–– CoOp/
|   |   |–– vit_b16_ctxv1_16shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
|   |   |–– vit_b16_ctxv1_8shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```

To calculate the average results for the folder `vit_b16_ep50_ctxv1_16shots/nctx16_cscFalse_ctpend/`, you can run

```bash
python parse_test_res.py output/imagenet/CoOp/vit_b16_ep50_ctxv1_16shots/nctx16_cscFalse_ctpend
```

Then, you will see something like this in your terminal

```bash
Parsing files in output/imagenet/CoOp/vit_b16_ep50_ctxv1_16shots/nctx16_cscFalse_ctpend
file: output/imagenet/CoOp/vit_b16_ep50_ctxv1_16shots/nctx16_cscFalse_ctpend/seed1/log.txt. accuracy: 71.80%. 
file: output/imagenet/CoOp/vit_b16_ep50_ctxv1_16shots/nctx16_cscFalse_ctpend/seed2/log.txt. accuracy: 71.40%. 
file: output/imagenet/CoOp/vit_b16_ep50_ctxv1_16shots/nctx16_cscFalse_ctpend/seed3/log.txt. accuracy: 71.60%. 
===
Summary of directory: output/imagenet/CoOp/vit_b16_ep50_ctxv1_16shots/nctx16_cscFalse_ctpend
* accuracy: 71.60% +- 0.16%
===
```




### Step 4: Linear Probe CLIP
Please move to [lpclip/](lpclip/).



## Acknowledgement
This repo is borrow from [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch). Thanks for their wonderful works.

