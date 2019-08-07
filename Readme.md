## About

The current repository is associated with the article ["Study of Deep Learning Techniques for
Side-Channel Analysis and Introduction to ASCAD Database"](https://eprint.iacr.org/2018/053.pdf) available on the [eprints](https://eprint.iacr.org).

Databases, Neural Networks models as well as scripts are provided here as a complementary
material to the article: please refer to it for various explanations and details about SCA and Deep Learning.

## The ASCAD database

**ASCAD** (ANSSI SCA Database) is a set of databases that aims at providing a benchmarking reference for the SCA community: the purpose is to have something similar to the [MNIST database](http://yann.lecun.com/exdb/mnist/)  that the Machine Learning community has been using for quite a while now to evaluate classification algorithms performances.

Several databases are available, depending on the underlying implementation and architecture. More information is available in the corresponding folders:
* [First-order boolean masked AES implementation on an ATMEGA8515](./ATMEGA_AES_v1)


## Copyright and license
Copyright (C) 2018, ANSSI and CEA

The databases, the Deep Learning models and the companion python  scripts of this repository are placed into the public domain.

## Authors

  * Ryad BENADJILA (<mailto:ryad.benadjila@ssi.gouv.fr>)
  * Eleonora CAGLI (<mailto:eleonora.cagli@cea.fr>)
  * Cécile DUMAS (<mailto:cecile.dumas@cea.fr>)
  * Emmanuel PROUFF (<mailto:emmanuel.prouff@ssi.gouv.fr>)
  * Rémi STRULLU (<mailto:remi.strullu@ssi.gouv.fr>)
  * Adrian THILLARD (<mailto:adrian.thillard@ssi.gouv.fr>)

## Acknowledgements

This work has been partially funded through the H2020 project REASSURE.

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
| ATMEGA boolean masked AES | fixed key    | Power (Icc) | [link](./ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/Readme.md) |
| ATMEGA boolean masked AES | variable key | Power (Icc) | [link](./ASCAD/ATMEGA_AES_v1/ATM_AES_v1_variable_key/Readme.md) |

3. Now you should be able to use the provided python scripts. If you have the `pip` Python package manager installed, getting the scripts dependencies is as simple as:
```
pip install keras numpy h5py matplotlib tensorflow tqdm
```

​		for Python 2, or:

```
pip3 install keras numpy h5py matplotlib tensorflow tqdm
```

​		for Python 3.

​		**Remark:** if using Python 3.7 version, we recommend to install version **1.5.0** of tensorflow (some incompatibility issues might arise otherwise ...).


​		For GPU acceleration, you might also want to install `tensorflow-gpu`:

```python
pip install tensorflow-gpu
```

​		ou 

```
pip3 install tensorflow-gpu
```



## <a name="ascad-companion-scripts"></a> ASCAD companion scripts

### Required Python packages

In order to use ASCAD companion scripts, here is the list of dependencies that need to be installed in your python setup:

  * The `h5py` HDF5 library ([http://www.h5py.org/](http://www.h5py.org/)).
  * The `keras` Machine Learning library ([https://keras.io/](https://keras.io/), tests have been performed on version **2.2.1**).
  * The `numpy` scientific computing library ([http://www.numpy.org/](http://www.numpy.org/)).
  * The `matplotlib` plotting library ([https://matplotlib.org/](https://matplotlib.org/)).
  *  The `TensorFlow` Deep Learning library for fast numerical computing created and released by Google ([https://www.tensorflow.org/](https://www.tensorflow.org/), tests have been performed on version **1.14** for Linux and on version **1.5.0** for Windows).

Note that these libraries are generally packaged in most of Linux distributions, and/or are available through the `pip` Python package manager. The case of the `keras` library is a bit special since many backends can be used (TensorFlow, Theano, ...) and depending on the target platform, CPU or GPU acceleration may be configured and used or not. For ASCAD scripts, we strongly suggest (specifically for the  profiling/training phase) to use a GPU backed configuration. Configuring `keras` backends and GPU acceleration won't be detailed here: please refer to [this Keras](https://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu)  and [this TensorFlow](https://www.tensorflow.org/versions/r0.12/how_tos/using_gpu/)  resources for more details on the topic (you will also certainly need to handle [Nvidia CUDA drivers](https://developer.nvidia.com/cuda-downloads) and libraries for you platform).

Finally, the scripts should work with Python 2 (except on WINDOWS) as well as Python 3.

We propose hereafter a high-level description of the proposed scripts. The default parameters of these scripts vary with the downloaded campaign, and are provided as a file in the corresponding folders:

| Implementation             | Campaign        | Platform  														| Link  																					 |
| :------------------------: | :-------------: | :-------------------------------------------------------------------------------------: | --------------------------------------------------------------------------------------- |
| ATMEGA boolean masked AES  | fixed key | Linux | [link](./ATMEGA_AES_v1/ATM_AES_v1_fixed_key/)    |
| ATMEGA boolean masked AES  | variable key | Linux | [link](./ATMEGA_AES_v1/ATM_AES_v1_variable_key/) |

Every script can be launched using the corresponding parameter file:
<pre>
$ python ASCAD_generate.py path_to_parameters_file
$ python ASCAD_train_models.py path_to_parameters_file
$ python ASCAD_test_models.py path_to_parameters_file
</pre>

If no parameter file is provided, the scripts run themselves with some default parameters.
<pre>
$ python ASCAD_generate.py
$ python ASCAD_train_models.py 
$ python ASCAD_test_models.py
</pre>

It is easy to run these scripts on custom parameters, either by modifying the default values within the script, or by creating a new parameter file.

### <a name="ascad-generation"></a> ASCAD generation
The [ASCAD_generate.py](ASCAD_generate.py) script is used to generate ASCAD databases from any of the available raw traces database. 

This script takes as an argument the name of a file containing a python dict with the following keys:
  * `traces_file`: this is the file name of the HDF5 raw traces with metadata database
  * `labeled_traces_file`: this is the name of the HDF5 output file.
  * `profiling_index`: this is a list corresponding to the index of profiling traces
  * `attack_index`: this is a list corresponding to the index of attack traces
  * `target_points`: this is the list of points of interest to extract from the traces
  * `profiling_desync`: this is the maximum desychronization applied to the profiling original traces, following uniformly randomly chosen values below this maximum for each trace.
  * `attack_desync`: this is the maximum desychronization applied to the attack original traces, following uniformly randomly chosen values below this maximum for each trace.

The `labelize` function is also of interest in the script: tuning it enables to generate databases that focus on other leaking spots of the masked AES (say byte 5 of the first round, byte 10 of the second round, and so on ...).

By tuning all these parameters, one is able to **generate multiple ASCAD databases** specialized in various values of interest, with customized desynchronization as well as customized profiling and attacking traces.

### Testing the trained models

The trained models can be tested using the [ASCAD_test_models.py](ASCAD_test_models.py) script.

The script computes the **ranking** of the real key byte among the 256 possible candidate bytes depending on the number of attack traces the trained model takes as input for prediction: this is a classical classification algorithm efficiency check in SCA (see the article for a more formal definition of the keys ranking).
The evolution of the rank with respect to the number of traces is plotted using `matplotlib`.

This script takes as an argument the name of a file containing a python dict with the following keys:

* `model_file`: this is an already trained model HDF5 file.
* `ascad_database`: this is an ASCAD database one wants to check the trained model on.
* `num_traces`: this is the maximum number of traces to process.

### Training the models
The trained CNNs and MLPs that we provide are all derived from one CNN architecture and one MLP architecture with architectural hyper-parameters discussed in the article  [ASCAD paper](https://eprint.iacr.org/2018/053.pdf).

We provide the [ASCAD_train_models.py](ASCAD_train_models.py) script in order to train the models with the training hyper-parameters that we explore and analyze in [ASCAD paper](https://eprint.iacr.org/2018/053.pdf).

This script takes as an argument the name of a file containing a python dict with the following keys:
* `ascad_database`: this is an ASCAD database one wants to use for the model training.
* `training_model`: this is the HDF5 file where the trained model is scheduled to be saved.
* `network_type`: this is the type of network of the model. Currently, only two types of model are supported by the script: 
  1. `mlp`: this is the multi-layer perceptron topology described in [ASCAD paper](https://eprint.iacr.org/2018/053.pdf);
  
  2. `cnn`: this is the convolutional neural network topology described in  [ASCAD paper](https://eprint.iacr.org/2018/053.pdf);
  
  3. `cnn2`: this is the convolutional neural network topology described in  [ASCAD paper](https://eprint.iacr.org/2018/053.pdf) adapted to the format of the traces in the "variable key" campaign [link](./ATMEGA_AES_v1/ATM_AES_v1_variable_key/Readme.md). 
  
     **It may happen that the script cannot be run directly with the training model; before, the field *input_dim* of the model description in file `ASCAD_train_models.py` must be adapted to exactly correspond to the dimension of the traces used for the training (e.g. by changing it from 1400 to 700 or vice-versa).**
* `epochs`: this is the number of epochs used for the training.
* `batch_size`: this is the size of the batch used for training.

## <a name="ascad-tests">







