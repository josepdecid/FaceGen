# FaceGen using GAN: Generative Adversarial Networks

##### Authors:
- Josep de Cid Rodríguez
- Gonzalo Recio Domènech
- Federico Loyola

## Requirements

The main requirement in this python is to have Python3.7.X installed.
It may come already installed in most devices. Otherwise, just go to their
[website](https://www.python.org/downloads/release/python-376/) and download the specific version for your OS.

The second important tool is [Pipenv](https://pipenv-fork.readthedocs.io/) a Python library for managing library
dependencies in virtual environments. The main idea behind using this tool is to be
easily able to track the packages used in our project in a centralized way, being able
to update, install or manage them for any purpose in a few simple commands.

Once Pipenv is installed, run in this folder the following command:

```shell
pipenv --python 3.7
pipenv install
```

which will create the virtual environment and install all the requirements specified in `Pipfile`.

### Usage

All commands can be run without entering the virtual environment shell by typing
`pipenv run` as a prefix before the `python` command. However, for simplicity,
we log into the environment shell. A prefix with the project name appearing before
each line of command shows its proper working:

```bash
> pipenv shell
(FaceGen) >
```

## Datasets

The used datasets are. Download and put them in `/datasets` folder.

- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [UTKFace](https://susanqq.github.io/UTKFace/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Environment variables

Copy the `.env_template` to a `.env` file and fill the specified environment variables required.

- `CIFAR_PATH`: Path of CIFAR10 dataset (required only if training with GA).
- `DATASET_PATH`: Path of the faces dataset (CelabA, UTKFace or any other dataset to run).
- `LOG_DIR`: Directory to save the logs for Tensorboard visualization.
- `CKPT_DIR`: Directory to save PyTorch models checkpoints.

This would be an example of `.env`:

```bash
CIFAR_PATH=$PROJECTS/FaceGen/datasets/CIFAR
DATASET_PATH=$PROJECTS/FaceGen/datasets/UTKFace
LOG_DIR=$PROJECTS/FaceGen/runs
CKPT_DIR=$PROJECTS/FaceGen/checkpoints
```

## Train the model

To train the model, simply run the following command:

```bash
(FaceGen) > python main.py {ALG}
```

where `ALG` can take the value of `GA` for Genetic Algorithms, `VAE` for Variatonal Auto-encoders and `GAN` for Generative Adversarial Networks.

If you want to ensure reproducibility, a manual `--seed` must be introduced in the command line:

```bash
(FaceGen) > python main.py GAN --seed 42
```

If we have an available GPU card that supports CUDA, it will be training the model using the GPU.

## Generate new samples from a pre-trained model

There are 4 different pre-trained models in this [MEGA folder](https://mega.nz/#F!bNVDGIoD!y2Bcy0BKfkwkR3KqdHxi7A).

To generate new samples, we can use the same script, just indicating the pre-trained model
path and the number of samples to generate using the `--generate` flag.

The following command generates 5 samples using the GAN model checkpoint named `model_42` which may be a file with `.pt`
extension located in the folder `ENV['CKPT_DIR']`. Results will be stored in `src/generated_samples`.

```bash
(FaceGen) > python main.py GAN --generate model_42 5
```

## Results Samples

Samples evolution along training process checkpoints from *VAE* with *UTKFace* dataset:
![VAE Generations](samples/VAE_UTKFace_epochs_evolution.gif)

Samples evolution along training process checkpoints from *VAE* with *CelebA* dataset:
![VAE Generations](samples/VAE_CelebA_epochs_evolution.gif)

Samples evolution along training process checkpoints from *GAN* with *UTKFace* dataset:
![VAE Generations](samples/GAN_UTKFace_epochs_evolution.gif)

Samples evolution along training process checkpoints from *GAN* with *CelebA* dataset:
![VAE Generations](samples/GAN_CelebA_epochs_evolution.gif)
