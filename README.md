## Introduction
This is our implementation of our paper *Text-Prompted Prompt Generator with Uncertainty Regularization for Rehearsal-Free Class-Incremental Learning*.

**TL;DR**: A generated prompt method for rehearsal-free class-incremental learning.

**Abstract**:
Prompt methods are a kind of popular methods for continual learning, by training a tiny collection of parameters based on frozen networks pre-trained on large-scale datasets, to adapt the model to sequential tasks. However, most of prompt methods encounters serious dependence on the delicately designed prompt pools and lead to two shortcomings: prompt inconsistency between training and inference, and prompt selection mismatch during inference. Benefiting from the uniqueness of language description and powerful vision transformers, we propose a Text-Prompted Prompt Generator Network (TPPNet), which designs a text-prompted prompt (TPP) generator by encapsulating the pre-trained text embeddings into the visual class token, yielding versatile TPP for resisting the shortcomings of previous methods. Typically, the versatile TPP exhibits three properties: (a) Expressiveness: TPP exhibits flexible expression for intra-task images, by incorporating prompted text encoder with text prompts; (b) Inter-task compatibleness: TPP equips the visual image token with the text embeddings of both old and current tasks, and absorbs old knowledge to shrink the prompt inconsistency and improve its anti-forgetting ability; (c) prompt-query avoidance: TPPNet avoids the prompt query process by generating instance-level prompts, and effectively handles the prompt selection mismatch issue during inference. We conduct experiments on four datasets, and the results show that TPPNet outperforms or is comparable with the state-of-the-art-methods for rehearsal-free class-incremental learning tasks. 


## Dependencies
- torch==1.8.1+cu111
- torchvision==0.9.1+cu111
- torchaudio==0.8.1
- numpy==1.21.2
- tqdm==4.66.3
- timm==0.6.11
- scipy==1.7.1



## Usage

##### 1. Install dependencies
First we recommend to create a conda environment with all the required packages by using the following command.
```
conda env create -f environment.yml
```
This command creates a conda environment named `TPPNet`. You can activate the conda environment with the following command:
```
conda activate TPPNet
```
In the following sections, we assume that you use this conda environment or you manually install the required packages.
Note that you may need to adapt the‘environment.yml/requirements.txt’files to your infrastructure. The configuration of these files was tested on Linux Platform with a GPU (RTX3090).
If you see the following error, you may need to install a PyTorch package compatible with your infrastructure.
```
RuntimeError: No HIP GPUs are available or ImportError: libtinfo.so.5: cannot open shared object file: No such file or directory
```
For example if your infrastructure only supports CUDA == 11.1, you may need to install the PyTorch package using CUDA11.1.
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

##### 2. Run code
- CIFAR-100
    ```
    python main.py --config=./exps/cifar100_vit.json
    ```

- ImageNet-R
    ```
    python main.py --config=./exps/imagenetr.json
    ```

- CUB200
    ```
    python main.py --config=./exps/cub200.json
    ```

- EuroSAT
    ```
    python main.py --config=./exps/eurosat.json
    ```

## Parameters

| Parameter         |           Description                       | 
|-------------------|---------------------------------------------|
| dataset              |   Dataset to use                            |
| increment           |   Number of classes learned for each task |
| APG_MLP_num         |   The number of MLPs in the generator |
| APG_num_heads         |   Number of self attention heads in the generator |
| APG_more_prompts       |   The number of generated prompts  |
| APG_attn_depth    |   The number of attention layers in the generator |
| device              |   GPU device ID (default: 0)                |
| batch_size        |   batch size for training    |
| epochs            |   epochs                    |

## 
