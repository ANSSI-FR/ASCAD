## About

**ASCAD** (ANSSI SCA Database) is a set of databases that aims at providing a benchmarking reference for the SCA community: the purpose is to have something similar to the [MNIST database](http://yann.lecun.com/exdb/mnist/) that the Machine Learning community has been using for quite a while now to evaluate classification algorithms performance.

This repository provides scripts and Deep Learning models that demonstrate the efficiency of Deep Learning for SCA.

Several databases are available, depending on the underlying implementation and architecture. More information is available in the corresponding folders:
* [First-order boolean masked AES implementation on an ATMEGA8515](./ATMEGA_AES_v1)
* [Affine masked AES implementation on an STM23](./STM32_AES_v2)

## Copyright and license
Copyright (C) 2021, ANSSI and CEA

The databases, the Deep Learning models and the companion python scripts of this repository are placed under the BSD licence.
Please check the [LICENSE](LICENSE) file for more information.

## <a name="getting-ascad"> Getting the ASCAD databases and the trained models 

### Quick start guide

The scripts and the data are split in two places mainly because git is not suited for large files.

In order to get everything up and running, here are the steps to follow (we provide the steps using a Unix shell syntax, but you can adapt this and use your favorite shell of course):

1. Clone the current repository to get the scripts:

```
git clone https://github.com/ANSSI-FR/ASCAD.git
```

2. Click on the link corresponding to the chosen campaign and follow the instructions to download and unpack the database.

| Implementation | Campaign     | Type        | Link  |
| ------------------------ |:-------------:| -----: | :----: |
| ATMEGA boolean masked AES | fixed key    | Power (Icc) | [link](./ATMEGA_AES_v1/ATM_AES_v1_fixed_key/Readme.md) |
| ATMEGA boolean masked AES | variable key | Power (Icc) | [link](./ATMEGA_AES_v1/ATM_AES_v1_variable_key/Readme.md) |
| STM32  affine masked AES  | variable key | Power (Icc) | [link](./STM32_AES_v2/Readme.md) |

3. Install the last version of Tensorflow 2 for your platform. Some versions of Tensorflow require a specific version of CUDA, up-to-date and detailed information on the installation can be found here: [https://www.tensorflow.org/install](https://www.tensorflow.org/install). You will also need Keras as the Tensorflow front end API companion. 

4. Now you should be able to use the provided python scripts. If you have the `pip` Python package manager installed, getting the scripts dependencies is as simple as:
```
pip install numpy h5py matplotlib tqdm
```
Our scripts **now rely on Tensorflow 2**, therefore **we only support Python 3**. If you want to continue to use Python 2, please refer to a previous version of the repository, for example
checkouting commit [30f65bb](https://github.com/ANSSI-FR/ASCAD/commit/30f65bb3279e949d74b7296ceda7f455fb70d591). However you will not be able to run the scripts on our recent databases (`STM32_AES_v2` and later).



## <a name="ascad-companion-scripts"></a> ASCAD companion scripts

### Required Python packages

In order to use ASCAD companion scripts, here is the list of dependencies that need to be installed in your python setup:

  * The `h5py` HDF5 library ([http://www.h5py.org/](http://www.h5py.org/)).
  * The `numpy` scientific computing library ([http://www.numpy.org/](http://www.numpy.org/)).
  * The `matplotlib` plotting library ([https://matplotlib.org/](https://matplotlib.org/)).
  * The `tensorflow 2` Deep Learning library for fast numerical computing created and released by Google ([https://www.tensorflow.org/](https://www.tensorflow.org/), tests have been performed on version **2.3.0** for Linux).
  * The `keras` Python Deep Learning API library ([https://keras.io/](https://keras.io/), tests have been performed on version **2.4.3** for Linux).

Note that these libraries are generally packaged in most of Linux distributions, and/or are available through the `pip` Python package manager. The case of the `tensorflow` library is a bit special since depending on the target platform, CPU or GPU acceleration may be configured and used or not. For ASCAD scripts, we strongly suggest (specifically for the  profiling/training phase) to use a GPU backed configuration. Configuring `tensorflow`  and GPU acceleration won't be detailed here: please refer to [this](https://www.tensorflow.org/install)  and [this](https://www.tensorflow.org/install/gpu/) resources for more details on the topic (you will also certainly need to handle [Nvidia CUDA drivers](https://developer.nvidia.com/cuda-downloads) and libraries for you platform).

Finally, please note that the scripts **only work with Python 3** since we rely on tensorflow 2.

We propose hereafter a high-level description of the proposed scripts. The default parameters of these scripts vary with the downloaded campaign, and are provided as a file in the corresponding folders:

| Implementation             | Campaign        | Platform  														| Link  																					 |
| :------------------------: | :-------------: | :-------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------- |
| ATMEGA boolean masked AES  | fixed key | Linux | [link](./ATMEGA_AES_v1/ATM_AES_v1_fixed_key/)    |
| ATMEGA boolean masked AES  | variable key | Linux | [link](./ATMEGA_AES_v1/ATM_AES_v1_variable_key/) |
| STM32 affine masked AES    | variable key | Linux | [link](./STM32_AES_v2/) |

Every script can be launched using the corresponding parameter file:
<pre>
$ python ASCAD_generate.py path_to_parameters_file
$ python ASCAD_train_models.py path_to_parameters_file
$ python ASCAD_test_models.py path_to_parameters_file
</pre>

It is easy to run these scripts on custom parameters, either by modifying the default values within the script, or by creating a new parameter file.

### <a name="ascad-generation"></a> ASCAD generation
The [ASCAD_generate.py](ASCAD_generate.py) script is used to generate ASCAD databases from any of the available raw traces database. 

This script takes as an argument the name of a file containing a python dict with the following keys:
  * `traces_file`: this is the file name of the HDF5 raw traces with metadata database. Use this argument if all the traces are contained in a single file.
  * (optional) `files_splitted` : set this option to 1 if the HDF5 raw traces are splitted in different files.
  * (optional) `traces_files_list` : this is the list of the HDF5 raw traces if the `files_splitted` option is set to 1. In this case `traces_file` argument is no more required.
  * (optional) `multilabel` : set this option to 1 to get a multiclassified dataset in the same vein that ASCADv2 attacks.
  * `labeled_traces_file`: this is the name of the HDF5 output file.
  * `profiling_index`: this is a list corresponding to the index of profiling traces.
  * `attack_index`: this is a list corresponding to the index of attack traces.
  * `target_points`: this is the list of points of interest to extract from the traces.
  * `profiling_desync`: this is the maximum desychronization applied to the profiling original traces, following uniformly randomly chosen values below this maximum for each trace.
  * `attack_desync`: this is the maximum desychronization applied to the attack original traces, following uniformly randomly chosen values below this maximum for each trace.

The `labelize` and `multilabelize` functions are also of interest in the script: tuning it enables to generate databases that focus on other leaking spots of the masked AES (say byte 5 of the first round, byte 10 of the second round, and so on ...).

By tuning all these parameters, one is able to **generate multiple ASCAD databases** specialized in various values of interest, with customized desynchronization as well as customized profiling and attacking traces.

### Testing the trained models

The trained models can be tested using the [ASCAD_test_models.py](ASCAD_test_models.py) script.

The script computes the **ranking** of the real key byte among the 256 possible candidate bytes depending on the number of attack traces the trained model takes as input for prediction: this is a classical classification algorithm efficiency check in SCA (see the article ["Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database"](https://eprint.iacr.org/2018/053.pdf) for a more formal definition of the keys ranking).
The evolution of the rank with respect to the number of traces is plotted using `matplotlib`.

This script takes as an argument the name of a file containing a python dict with the following keys:

* `model_file`: this is an already trained model HDF5 file.
* `ascad_database`: this is an ASCAD database one wants to check the trained model on.
* `num_traces`: this is the maximum number of traces to process.
* (optional) `simulated_key` : if the part of the dataset used during the attack step has not a constant key, this option simulates a constant key equal to 0 when the rank is computed (the new plaintext is equal to the previous plaintext xor the current key).
* (optional) `target_byte` : this is the index of the target byte during the attack. Default value is equal to 2 for ASCADv1 retrocompatiblity.
* (optional) `multilabel` : perform a multilabel attack by recombining the probabilty to compute Pr(Sbox|t). If set to 1, the permindices of the shuffling are taken into account in the recombination. If set to 2, the permindices are not taken into account. If set to 0 (default value), it performs a single label computation.
* (optional) `save_file` : if specified, it saves the plot in a file with name `save_file`.


### Training the models
<!-- The trained CNNs and MLPs that we provide are all derived from one CNN architecture and one MLP architecture with architectural hyper-parameters discussed in the article  [ASCAD paper](https://eprint.iacr.org/2018/053.pdf). -->

We provide the [ASCAD_train_models.py](ASCAD_train_models.py) script in order to train the models. This script takes as an argument the name of a file containing a python dict with the following keys:

* `ascad_database`: this is an ASCAD database one wants to use for the model training.
* `training_model`: this is the HDF5 file where the trained model is scheduled to be saved.
* `network_type`: this is the type of network of the model. Currently, three types of model are supported by the script: 
  1. `mlp`: this is the multi-layer perceptron topology described in [ASCAD paper](https://eprint.iacr.org/2018/053.pdf);
  
  2. `cnn`: this is the convolutional neural network topology described in  [ASCAD paper](https://eprint.iacr.org/2018/053.pdf);
  
  3. `cnn2`: this is the convolutional neural network topology described in  [ASCAD paper](https://eprint.iacr.org/2018/053.pdf) adapted to the format of the traces in the "variable key" campaign [link](./ATMEGA_AES_v1/ATM_AES_v1_variable_key/Readme.md). 
  
  4. `multi_resnet`: this is the multiclassification ResNet model described during the "GDR SoC2 et Sécurité informatique" ([video](https://mediacenter3.univ-st-etienne.fr/videos/?video=MEDIA201125165945975)). This model requires the knowledge of the permutation indices of the shuffling operation during the training step.

  5. `multi_resnet_without_permind`: this is the multiclassification ResNet model described during the "GDR SoC2 et Sécurité informatique" ([video](https://mediacenter3.univ-st-etienne.fr/videos/?video=MEDIA201125165945975)). This model does not take the shuffling into account.

* `epochs`: this is the number of epochs used for the training.
* `batch_size`: this is the size of the batch used for training.
* (optional) `train_len`: this is the number of traces of the training dataset that are used to train the model. This number shall be less than the total number of traces in the training dataset.
* (optional) `validation_split`: this is the fraction of the training dataset to use as a validation dataset during the training step.
* (optional) `multilabel`: this option shall be set to a non null value when a multilabel model is trained. Set this value to 1 to train the `multi_resnet` model, and to 2 to train the `multi_resnet_without_permind` model.
* (optional) `early_stopping`: if this option is set to a non null value, the model is trained with an early stopping strategy on the validation cross-entropy. The size of the validation dataset is controlled with the `validation_split` option. If the `validation_split` option was not previously set, then its default value is equal to 10 pcts of the training dataset.
## <a name="ascad-tests">







