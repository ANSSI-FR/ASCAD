## About

The current repository is associated with the article ["Study of Deep Learning Techniques for
Side-Channel Analysis and Introduction to ASCAD Database"](https://eprint.iacr.org/2018/053.pdf)
available on the [eprints](https://eprint.iacr.org).

Databases, Neural Networks models as well as scripts are provided here as a complementary
material to the article: please refer to it for various explanations and
details about SCA and Deep Learning.

## Copyright and license
Copyright (C) 2018, ANSSI and CEA

The databases, the Deep Learning models and the companion python 
scripts of this repository are placed into the public domain.

## Authors

  * Ryad BENADJILA (<mailto:ryad.benadjila@ssi.gouv.fr>)
  * Eleonora CAGLI (<mailto:eleonora.cagli@cea.fr>)
  * Cécile DUMAS (<mailto:cecile.dumas@cea.fr>)
  * Emmanuel PROUFF (<mailto:emmanuel.prouff@ssi.gouv.fr>)
  * Rémi STRULLU (<mailto:remi.strullu@ssi.gouv.fr>)

## Acknowledgements

This work has been partially funded through the H2020 project REASSURE.
   
## <a name="getting-ascad"> Getting the ASCAD databases and the trained models 

### Quick start guide

The scripts and the data are split in two places mainly
because git is not suited for large files.

In order to get everything up and running, here are the steps to follow
(we provide the steps using a Unix shell syntax, but you can adapt this
and use your favorite shell of course):

1. Clone the current repository to get the scripts:
<pre>
$ git clone https://github.com/ANSSI-FR/ASCAD.git
</pre>

2. In the new `ASCAD` folder, download and decompress the data
package with the raw data by using:
<pre>
$ cd ASCAD
$ wget https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip
$ unzip ASCAD_data.zip
</pre>

Please be aware that this last step should 
**download around 4.2 GB**, and the decompression will
generate around **7.3 GB of useful data**.

Now you should be able to use the provided python scripts.
If you have the `pip` Python package manager installed, getting 
the scripts dependencies is as simple as:
<pre>
$ pip install keras numpy h5py matplotlib tensorflow
for Python 2, or:
$ pip3 install keras numpy h5py matplotlib tensorflow
for Python 3.
</pre>

For GPU acceleration, you might also want to install `tensorflow-gpu`:
<pre>
$ pip install tensorflow-gpu
for Python 2, or:
$ pip3 install tensorflow-gpu
for Python3.
</pre>

A quick test to check that everything is properly installed and working fine 
can be done by launching the [ASCAD_test_models.py](ASCAD_test_models.py) script
from the main ASCAD folder:
<pre>
$ python ASCAD_test_models.py
</pre>

More details are provided in the [dedicated section](#ascad-companion-scripts).

### Raw data files hashes

The current repository **only contains scripts**: the raw data that
are manipulated by these scripts can be found here:
[ASCAD_data](http://data.ascad-databases.ovh/ASCAD_data.zip).

The zip file SHA-256 hash value is:
<hr>

**ASCAD_data.zip**
`a6884faf97133f9397aeb1af247dc71ab7616f3c181190f127ea4c474a0ad72c`

</hr>

We also provide the SHA-256 hash values of the sub-files when this
zip archive is decompressed:

<hr>

**ASCAD_databases/ASCAD.h5:**
`f56625977fb6db8075ab620b1f3ef49a2a349ae75511097505855376e9684f91`
**ASCAD_databases/ASCAD_desync50.h5:**
`8716a01d4aea2df0650636504803af57dd597623854facfa75beae5a563c0937`
**ASCAD_databases/ASCAD_desync100.h5:**
`f6b9e967af287e82f0a152320e58f8f0ded35cd74d499b5f7b1505a5ce338b8e`
**ASCAD_databases/ATMega8515_raw_traces.h5:**
`51e722f6c63a590ce2c4633c9a9534e8e1b22a9cde8e4532e32c11ac089d4625`
<hr>

**ASCAD_trained_models/mlp_best_ascad_desync0_node200_layernb6_epochs200_classes256_batchsize100.h5:**
`d97a6e0f742744d0854752fce506b4a0612e0b86d0ec81a1144aada4b6fb35a3`
**ASCAD_trained_models/mlp_best_ascad_desync50_node200_layernb6_epochs200_classes256_batchsize100.h5**
`582a590c69df625fd072f837c98e147a83e4e20e04465ff48ca233b02bc75925`
**ASCAD_trained_models/mlp_best_ascad_desync100_node200_layernb6_epochs200_classes256_batchsize100.h5:**
`9f4d761197b91b135ba24dd84104752b7e32f192ceed338c26ddba08725663a9`
**ASCAD_trained_models/cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5:**
`11ff0613d71ccd026751cb90c2043aff24f98adb769cb7467e9daf47567645be`
**ASCAD_trained_models/cnn_best_ascad_desync50_epochs75_classes256_batchsize200.h5:**
`be9045672095a094d70d2ee1f5a76277cab6a902c51e4ebf769282f464828a11`
**ASCAD_trained_models/cnn_best_ascad_desync100_epochs75_classes256_batchsize200.h5:**
`866d3ea0e357e09ff30fdc9c39b6ef3096262c50cebd42018a119b1190339fcc`
<hr>

> **WARNING: all the paths and examples that are provided below suppose that you have
downloaded and decompressed the raw data file as explained in [the previous section](#getting-ascad).**

<hr>

## The ATMega8515 SCA traces database

ANSSI has provided source code implementations of two **masked AES** on the ATMega8515
MCU target, which can be found on the following github 
repository: [ANSSI-FR/secAES-ATmega8515](https://github.com/ANSSI-FR/secAES-ATmega8515). ATMega8515 uses external clocking: the acquisitions have been performed using a smartcard reader providing a **4 MHz clock**, compatible with the ISO7816-3 standard default values.

The ASCAD databases correspond to the first version (v1) of the masked AES (the second version has improved security and is currently kept for further studies). The v1 implementation allowed us to perform the acquisition of EM (ElectroMagnetic) measurements
in traces of 100,000 time samples (at 2 giga-samples per second) on an 
[ATMega8515](http://www.infinityusb.com/default.asp?show=store&ProductGrp=8) 
based [WB Electronics 64 Kbit ATMega chipcard](http://www.infinityusb.com/default.asp?show=store&ProductGrp=8) 
(see the [secAES-ATmega8515](https://github.com/ANSSI-FR/secAES-ATmega8515)
material for more information). The traces are synchronized, and no
specific hardware countermeasure has been activated on the ATMega8515.

An extract of 60,000 traces from the acquisition campaign has been compiled in one 
[HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file
of 5.6 GB named `ATMega8515_raw_traces.h5`. The structure of this HDF5 file is 
described in the article ["Study of Deep Learning Techniques for Side-Channel 
Analysis and Introduction to ASCAD Database"](https://eprint.iacr.org/2018/053.pdf).

## The ASCAD database

**ASCAD** (ANSSI SCA Database) is a set of databases that aims at providing
a benchmarking reference for the SCA community: the purpose is to
have something similar to the [MNIST database](http://yann.lecun.com/exdb/mnist/) 
that the Machine Learning community has been using for quite a
while now to evaluate classification algorithms performances.

The databases, which are HDF5 files, basically contain two labelled datasets:
  * A 50,000 traces **profiling dataset** that is used to train the (deep) Neural Networks models.
  * A 10,000  traces **attack dataset** that is used to check the performance of the trained models after the
profiling phase. 

The details of the ASCAD HDF5 structure are given in the article, as well as
a thorough discussion about the elements that need to
be addressed when applying Deep Learning techniques to SCA.

The ASCAD database is in fact extracted from the `ATMega8515_raw_traces.h5`
file containing raw traces: in order to avoid useless heavy
data processing, only the 700 samples of interest are kept
(these samples correspond to the time window of the leaking
operation under attack, see the article for details).

The [ASCAD_generate.py](ASCAD_generate.py) script is used to
generate ASCAD from the ATMega8515 original database. Actually, 
the repository contains three HDF5 ASCAD databases:
  * `ASCAD_data/ASCAD_databases/ASCAD.h5`: this corresponds to 
  the original traces extracted without modification.
  * ` ASCAD_data/ASCAD_databases/ASCAD_desync50.h5`: this
  contains traces desynchronized with a 50 samples maximum window.
  * `ASCAD_data/ASCAD_databases/ASCAD_desync100.h5`: this
  contains traces desynchronized with a 100 samples maximum window.

The traces **desynchronization** is simulated artificially (and can be tuned)
by the python script [ASCAD_generate.py](ASCAD_generate.py) that generates the database: 
this allows us to test the efficiency of Neural Networks against 
**jitter**. See the [section dedicated to the
generation script](#ascad-generation) for details on how to
customize the desynchronization parameter.


## The trained models

The best **trained CNN and MLP models** that we have obtained are provided in the
`ASCAD_data/ASCAD_trained_models/` folder. Six models have
been selected: best CNNs for desynchronizations 0, 50 and 100,
best MLPs for desynchronization values of 0, 50, and 100
time samples.

**WARNING**: these models are the best ones we have
obtained through the methodology described in the article.
We certainly **do not pretend** that they are the optimal models 
across all the possible ones. The main purpose of sharing
ASCAD is precisely to explore and evaluate new models.

We have performed our training sessions on two setups:
  * The first platform is composed of one gamer market **Nvidia GeForce GTX 1080 Ti**.
  * The second platform is composed of one professional computing market **Nvidia Tesla K80**.  

Both setups were running an Ubuntu 16.04 distro with Keras 2.1.1 and TensorFlow-GPU 1.2.1.
When using the GPU acceleration, the computation should not be very CPU and RAM intensive
(at most one CPU core work load and 1 to 2 GB of RAM).

See below for how to test these trained models.

## <a name="ascad-companion-scripts"></a> ASCAD companion scripts

### Required Python packages

In order to use ASCAD companion scripts, here is the list of
dependencies that need to be installed in your python
setup:

  * The `h5py` HDF5 library ([http://www.h5py.org/](http://www.h5py.org/)).
  * The `keras` Deep Learning library ([https://keras.io/](https://keras.io/), tests have been
  performed on version **2.2.1**).
  * The `numpy` scientific computing library ([http://www.numpy.org/](http://www.numpy.org/)).
  * The `matplotlib` plotting library ([https://matplotlib.org/](https://matplotlib.org/)).

Note that these libraries are generally packaged in most of Linux distributions,
and/or are available through the `pip` Python package manager. The case of the `keras`
library is a bit special since many backends can be used (TensorFlow, Theano, ...)
and depending on the target platform, CPU or GPU acceleration may be configured
and used or not. For ASCAD scripts, we strongly suggest (specifically for the 
profiling/training phase) to use a GPU backed configuration. Configuring
`keras` backends and GPU acceleration won't be detailed here: please refer
to [this Keras](https://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu) 
and [this TensorFlow](https://www.tensorflow.org/versions/r0.12/how_tos/using_gpu/) 
resources for more details on the topic (you will also certainly need to
handle [Nvidia CUDA drivers](https://developer.nvidia.com/cuda-downloads) 
and libraries for you platform).

Finally, the scripts should work with Python 2 as well as Python 3.

### <a name="ascad-generation"></a> ASCAD generation

The [ASCAD_generate.py](ASCAD_generate.py) script is used to generate ASCAD
databases from the ATMega8515 raw traces database. This script is versatile 
and the `extract_traces` function accepts some parameters:

<pre>
def extract_traces(traces_file, labeled_traces_file, profiling_index = [n for n in range(0, 50000)], attack_index = [n for n in range(50000, 60000)], target_points=[n for n in range(45400, 46100)], profiling_desync=0, attack_desync=0)
</pre>

  * `traces_file`: this is the file name of the HDF5 raw traces with metadata 
  database, `ATMega8515_raw_traces.h5` in our case.
  * `labeled_traces_file`: this is the name of the HDF5 output file.
  * `profiling_index`: this is a list corresponding to the index
  of profiling traces (default is [0 .. 49999])
  * `attack_index`: this is a list corresponding to the index of
  attack traces (default is [50000 .. 59999]).
  * `target_points`: this is the window of interest in the traces,
  default is [45400 .. 46099] since this is the leaking spot for
  the value of interest in the case described in the article.
  * `profiling_desync`: this is the maximum desychronization applied
  to the profiling original traces, following uniformly randomly chosen values below this
  maximum for each trace.
  * `attack_desync`: this is the maximum desychronization applied
  to the attack original traces, following uniformly randomly chosen values below this
  maximum for each trace.

The `labelize` function is also of interest in the script: tuning it enables to
generate databases that focus on other leaking spots of the masked AES (say 
byte 5 of the first round, byte 10 of the second round, and so on ...).

By tuning all these parameters, one is able to **generate multiple ASCAD databases**
specialized in various values of interest, with customized desynchronization
as well as customized profiling and attacking traces.

### Testing the trained models

The trained models in `ASCAD_data/ASCAD_trained_models/` can be tested using
the [ASCAD_test_models.py](ASCAD_test_models.py) script.

The script computes the **ranking** of the real key byte among the
256 possible candidate bytes depending on the number of attack traces
the trained model takes as input for prediction: this is a
classical classification algorithm efficiency check in SCA 
(see the article for a more formal definition of the rank).
The evolution of the rank with respect to the number of traces is
plotted using `matplotlib`.

Without any argument, the script will compute the rank on
all the trained models (CNN_Best for desynchronizations 
0, 50, 100 and MLP_Best for desynchronizations 0, 50, 100)
for 2000 traces. One can also modify this number of
traces with one argument:

<pre>
$ python ASCAD_test_models.py
or:
$ python ASCAD_test_models.py 5000 
</pre>


Optionally, the script takes two or three  arguments as inputs:
  * An already trained model HDF5 file (for instance those in the 
  `ASCAD_data/ASCAD_trained_models/` folder).
  * An ASCAD database one wants to check the trained model on.
  * A third optional parameter is the maximum number of attack traces
to process.

<pre>
$ python ASCAD_test_models.py ASCAD_data/ASCAD_train_models/cnn_best_ascad_desync0.h5 ASCAD_data/ASCAD_databases/ASCAD.h5
or:
$ python ASCAD_test_models.py ASCAD_data/ASCAD_trained_models/cnn_best_ascad_desync0.h5 ASCAD_data/ASCAD_databases/ASCAD.h5 5000
</pre>

The script uses the **attack traces set** in a hardcoded way (since
we want to check the model efficiency on traces that have not
been used for training). However, it is pretty straightforward
to tune it to compute the ranking on the ASCAD profiling traces
to further confirm the cross-validation results obtained in the 
article.

### Deep Learning with ASCAD: training the models

The six trained CNNs and MLPs that we provide are all derived from
one CNN architecture and one MLP architecture with architectural
hyper-parameters discussed in the article (the main difference is
the training that is performed on the three desynchronized {0, 50, 100}
ASCAD databases).

We provide the [ASCAD_train_models.py](ASCAD_train_models.py)
script in order to train the models with
the training hyper-parameters that we explore and analyze in 
the article: trained models should yield in similar
performances compared to what we provide in
`ASCAD_data/ASCAD_trained_models/`.

The training is performed on the 50,000 **profiling traces**, but
one can easily tune the script to modify this for
including other sets of traces if necessary.
