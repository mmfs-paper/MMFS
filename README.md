# MMFS 

Official PyTorch implementation of MMFS: Multi-Modal Face Stylization with a Generative Prior.

> **MMFS: Multi-Modal Face Stylization with a Generative Prior**,             
> Mengtian Li<sup>1</sup>, Yi Dong<sup>2</sup>, Minxuan Lin<sup>1</sup>, Haibin Huang<sup>1</sup>, Pengfei Wan<sup>1</sup>, Chongyang Ma<sup>1</sup>,    
> _<sup>1</sup>Kuaishou Technology, Beijing, China_  
> _<sup>2</sup>Tsinghua University, Beijing, China_  
> In: Pacific Graphics 2023 (**CGF**) 
> *[arXiv preprint](https://arxiv.org/abs/2305.18009)* 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/dongyi/MMFS)

<img width="1296" alt="image" src="https://github.com/dong-y21/MMFS/assets/106572559/d40e2d02-66d6-4ea7-bb43-d7f9d73ea5e1">




## Downloads

| Pretrained Models | Description | Link | Link |
| --- | --- | --- | --- |
|phase2_pretrain_90000.pth | pretrained phase2 model（Dataset AAHQ) | [![](https://img.shields.io/static/v1?message=Baidu%20Netdisk&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://pan.baidu.com/s/1rz7rPjngmdcL28sWwPJmKw?pwd=2xsg) | [![](https://img.shields.io/static/v1?message=Google%20Drive&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://drive.google.com/file/d/1jPXIR5UqkWS7chsMZ-SR-yAs0WIGCO3p/view?usp=drive_link)  
|phase3_pretrain_10000.pth | pretrained phase3 model (for phase4 fine-tuning) | [![](https://img.shields.io/static/v1?message=Baidu%20Netdisk&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://pan.baidu.com/s/1w2OLkAUSPQbwxXu_30naCw?pwd=ncm9) | [![](https://img.shields.io/static/v1?message=Google%20Drive&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://drive.google.com/file/d/12AfgFfOs8PjagtYwglmO9bquO2AEzPof/view?usp=drive_link)

| One-Shot Models | Img Prompt | Link | Link |
| --- | --- | --- | --- |
|example_ref01.pth | <img src="example/reference/01.png" width="200px"> | [![](https://img.shields.io/static/v1?message=Baidu%20Netdisk&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://pan.baidu.com/s/1S2YCXh14hLq2bILW3asmQw?pwd=wjmd) | [![](https://img.shields.io/static/v1?message=Google%20Drive&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://drive.google.com/file/d/1nip981zqzASsPu6EiRRXBYvHOAosqPMj/view?usp=drive_link)  
|example_ref02.pth | <img src="example/reference/02.png" width="200px"> | [![](https://img.shields.io/static/v1?message=Baidu%20Netdisk&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://pan.baidu.com/s/17uclEk1bPOmwjDDtU9rtuQ?pwd=qvjx) | [![](https://img.shields.io/static/v1?message=Google%20Drive&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://drive.google.com/file/d/1Lq1PqeHKWbNgoIFsCHCSzXPIKmHRDtlh/view?usp=drive_link)  
|example_ref03.pth | <img src="example/reference/03.png" width="200px"> | [![](https://img.shields.io/static/v1?message=Baidu%20Netdisk&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://pan.baidu.com/s/1ma6ueCq0o45mWEC8uSnecg?pwd=37md) | [![](https://img.shields.io/static/v1?message=Google%20Drive&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://drive.google.com/file/d/1UCBpnT7BC4fd1l7vu7YalyPz8ugPbom8/view?usp=drive_link)  
|example_ref04.pth | <img src="example/reference/04.png" width="200px"> | [![](https://img.shields.io/static/v1?message=Baidu%20Netdisk&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://pan.baidu.com/s/1Q60Jejc9EuE3lDr7-mPv1w?pwd=x8d4) | [![](https://img.shields.io/static/v1?message=Google%20Drive&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://drive.google.com/file/d/1qEjDFsX-z1anpDr54dP5VG2LmSS1DU3R/view?usp=drive_link)  

| Zero-Shot Models | Txt Prompt | Link | Link |
| --- | --- | --- | --- |
|pop_art.pth | pop art | [![](https://img.shields.io/static/v1?message=Baidu%20Netdisk&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://pan.baidu.com/s/1hkjJQrwIPHWEasZmL3aViA?pwd=4uxi) | [![](https://img.shields.io/static/v1?message=Google%20Drive&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://drive.google.com/file/d/17a0OJjF4PuSCIouDMnVuc5iiGLRPZhOx/view?usp=drive_link)  
|watercolor_painting.pth | watercolor painting | [![](https://img.shields.io/static/v1?message=Baidu%20Netdisk&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://pan.baidu.com/s/1kQHr0Plbcux9cZ9GOdfWNA?pwd=atve) | [![](https://img.shields.io/static/v1?message=Google%20Drive&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://drive.google.com/file/d/1QGgzsiXQgJt_gjRMFQbv5_qS0kgntzBV/view?usp=drive_link)  


## Fine-tuning

Fine-tuning for one-shot and zero-shot:
```bash
python train.py --cfg_file ./exp/sp2pII-phase4.yaml
```
- **Required Configs in [sp2pII-phase4.yaml](exp/sp2pII-phase4.yaml)**:   
***name:*** name of saving folder   
***pretrained_model:*** path to pretrained phase3 model   
**(one-shot only)** ***image_prompt:*** path to reference style image   
**(zero-shot only)** ***text_prompt:*** description of reference style   



## Reference
#### Multi-Model

- Stylize a folder:  
```bash
python test.py --cfg_file ./exp/sp2pII-phase2.yaml --test_folder path/to/your/test/folder 
--ckpt path/to/your/phase2/checkpoint --overwrite_output_dir path/to/save/your/results
```

- Stylize one image：
```bash
python test.py --cfg_file ./exp/sp2pII-phase2.yaml --test_img path/to/your/test/image
--ckpt path/to/your/phase2/checkpoint --overwrite_output_dir path/to/save/result
```

#### One-Shot
```bash
python test_sp2pII.py --ckpt path/to/your/phase4/checkpoint --in_folder path/to/your/test/folder 
--out_folder path/to/save/your/results --img_prompt path/to/reference/style/image --device "cpu or cuda:x"
```

#### Zero-Shot
```bash
python test_sp2pII.py --ckpt path/to/your/phase4/checkpoint --in_folder path/to/your/test/folder 
--out_folder path/to/save/your/results --txt_prompt "description of reference style" --device "cpu or cuda:x"
```
