# AugPrompt

This work proposes a new prompt learning framework named Augmented Vision-Guided Prompt Learning (AugPrompt). It performs multiple image augmentation operations on the image branch and constructs an Augmented Prompt through semantic mapping between the image and text branches (by Meta-Net). Utilizing the characteristics of the Text Encoder we found, it integrates the Context Prompt and Augmented Prompt to obtain a unified text feature, which is then used for contrastive learning with the original image's feature.



## 1. How to Install

### 1.1 Toolbox Dassl.pytorch

Firstly, the code is built on the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), thus we need to install the `dassl` environment. We recommend opening a terminal or command line window in the `AugPrompt`/ directory and executing the relevant commands.

```sh
# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

### 1.2 Other packages

After that, install a few more packages required by [CLIP](https://github.com/openai/CLIP) or augmentation operations. We recommend opening a terminal or command line window in the `AugPrompt`/ directory and executing the relevant commands.

```sh
cd CODE/

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt
```

### 1.3 Datasets

Follow [README.md](DATA/README.md) in `DATA/` to install the 9 datasets we used in Experiments.

Then, you are ready to go.



## 2. How to Run

The running scripts are provided in `CODE/scripts/cocoop/` (Since our AugPrompt method is implemented on [CoCoOp](https://github.com/KaiyangZhou/CoOp), thus we run the modified CoCoOp code directly.).

Make sure you change the path of `DATA` in scripts and run the commands under the directory `CODE/`.

### 2.1 Generalization From Base to New Classes

You will need both `CODE/scripts/cocoop/base2new_train.sh` and `CODE/scripts/cocoop/base2new_test.sh`. The former trains and evaluates the model on bash classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

`DATASET` takes as input a dataset name, like `caltech101` or `oxford_pets`. The valid names are the files' names in `CODE/configs/datasets/`.

Below we provide an example on how to train and evaluate the model on FGVC_Aircraft.

```bash
# seed=1
sh scripts/cocoop/base2new_train.sh fgvc_aircraft 1
sh scripts/cocoop/base2new_test.sh fgvc_aircraft 1

# or 
# seed=2
bash scripts/cocoop/base2new_train.sh fgvc_aircraft 2
bash scripts/cocoop/base2new_test.sh fgvc_aircraft 2
```

After you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on FGVC_Aircraft using the aforementioned commands, you would get results in `CODE/output`.

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– fgvc_aircraft/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c16_ep10_batch1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |–– train_base/
|   |   |–– fgvc_aircraft/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c16_ep10_batch1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
```

It's worth noting that, several configurations can be adjusted in the shell script for training models. Below is an overview of the key variables that you may need to modify:

- `DATA`: This is the root directory where the datasets are stored.
- `CFG`: Defines the model parameters for training. For example, `vit_b16_c16_ep10_batch1` specifies using ViT/B16 as the Image Encoder, a context length of 16, training for 10 epochs, and a batch size of 1. `ctxv1` indicates that the context prompt has a default initialization. All available `CFG yaml` configurations can be found in `CODE/configs/trainers/CoCoOp`.
- **SHOTS**: Specifies the number of shots for few-shot learning.
- **DIR**: The directory where the trained models and evaluation results are stored.
- **MODEL_DIR**: The path to read the trained models from.

### 2.2 Standard Few-shot learning

You will need to use `CODE/scripts/cocoop/all_train.sh`, which will train using all categories and return the accuracy on the test set. If you already have a model, you can directly use `CODE/scripts/cocoop/all_test.sh` to evaluate the model. 

Similar to the above, both scripts have two input arguments, i.e., `DATASET` and `SEED`. Below we provide an example of how to train and evaluate the model on OxfordPets.

```bash
# seed=1
sh scripts/cocoop/all_train.sh oxford_pets 1
sh scripts/cocoop/all_test.sh oxford_pets 1

# or 
# seed=2
bash scripts/cocoop/all_train.sh oxford_pets 2
bash scripts/cocoop/all_test.sh oxford_pets 2
```

You would get results in `CODE/output`.

```
output
|–– all/
|   |–– test/
|   |   |–– oxford_pets/
|   |   |   |–– CoCoOp/
|   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1_16shots/
|   |   |   |   |   |–– seed1/
|   |   |   |   |   |–– seed2/
|   |–– train/
|   |   |–– oxford_pets/
|   |   |   |–– CoCoOp/
|   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1_16shots/
|   |   |   |   |   |–– seed1/
|   |   |   |   |   |–– seed2/
```

Again, you can change some of the configuration to suit your needs.

### Selection of Augmentation Methods:

You can choose the desired augmentation methods from `AUGMENTED_TRANSFORMATIONS` in `CODE/trainers/cocoop.py`. There are a total of 10 augmentation methods available for selection.

```python
['HORIZONTAL_FLIP', 'ROTATION', 'TRANSLATE', 'SHEAR', 'RANDOM_PERSPECTIVE', 'CROP', 'RANDOM_ERASING', 'RANDOM_EQUALIZE', 'COLOR_JITTER', 'ANIME_GAN']
```

Additionally, you can modify the `INTENSITY` to control the random range of augmentation operations (It is recommended to set the `INTENSITY` to be less than 0.3). 



## Additional Notes:

1. **Number of Seeds**: The model is sensitive due to two reasons: 

   - the random initialization in few-shot learning,
   - the random initialization of all augmentation operations (within a certain range). 

   Therefore, it is necessary to experiment with multiple random seeds.

2. **Dataset Bug**: There is an issue with path handling for the DTD dataset in the Linux environment. Hence, we recommend running it in a Windows environment.
