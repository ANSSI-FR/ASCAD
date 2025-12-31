### ATMEGA8515, AES Boolean masking, fixed key, EM acquisition

## <a name="download"> Getting the ASCAD databases and the trained models 

In the new folder, download and decompress the data package with the raw data by using:
<pre>
$ cd ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/
$ wget https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e -O ASCAD_data.zip
$ unzip ASCAD_data.zip
</pre>

Please be aware that this last step should **download around 4.2 GB**, and the decompression will
generate around **7.3 GB of useful data**.

### Raw data files hashes
The zip file SHA-256 hash value is:
<hr>

**ASCAD_data.zip**
`a6884faf97133f9397aeb1af247dc71ab7616f3c181190f127ea4c474a0ad72c`

We also provide the SHA-256 hash values of the sub-files when this zip archive is decompressed:

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

## The ATMega8515 SCA traces databases

This database contains 60,000 traces from the acquisition campaign compiled in a [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file of 5.6 GB named `ATMega8515_raw_traces.h5`. The structure of this HDF5 file is described in the article ["Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database"](https://eprint.iacr.org/2018/053.pdf).

We emphasize that these traces are **synchronized**, and that the key is **fixed** for all the acquisitions.

## The ASCAD databases
The databases, which are HDF5 files, basically contain two labeled datasets:
  * A 50,000 traces **profiling dataset** that is used to train the (deep) Neural Networks models.
  * A 10,000 traces **attack dataset** that is used to check the performance of the trained models after the
profiling phase. 

The details of the ASCAD HDF5 structure are given in the article, as well as a thorough discussion about the elements that need to be addressed when applying Deep Learning techniques to SCA.

The ASCAD database is in fact extracted from the `ATMega8515_raw_traces.h5` file containing raw traces: in order to avoid useless heavy data processing, only the 700 samples of interest are kept (these samples correspond to the time window of the leaking operation under attack, see the article for details).

The [../ASCAD_generate.py](ASCAD_generate.py) script has been used to generate ASCAD from the `ATMega8515_raw_traces.h5` original database. Actually, 
the repository contains three HDF5 ASCAD databases:

  * `ASCAD_data/ASCAD_databases/ASCAD.h5`: this corresponds to 
    the original traces extracted without modification.
  * ` ASCAD_data/ASCAD_databases/ASCAD_desync50.h5`: this
    contains traces desynchronized with a 50 samples maximum window.
  * `ASCAD_data/ASCAD_databases/ASCAD_desync100.h5`: this
    contains traces desynchronized with a 100 samples maximum window.

The traces **desynchronization** has been simulated artificially (and can be tuned) by the python script [../ASCAD_generate.py](ASCAD_generate.py) that generates the database:  this allowed us to test the efficiency of Neural Networks against **jitter**. See the [section dedicated to the generation script](../../Readme.md) for details on how to customize the desynchronization parameter.


## The trained models 

The best **trained CNN and MLP models** that we have obtained are provided in the`ASCAD_data/ASCAD_trained_models/` folder. Six models have been selected: best CNNs for desynchronizations 0, 50 and 100, best MLPs for desynchronization values of 0, 50, and 100 time samples.

**WARNING**: these models are the best ones we have obtained through the methodology described in the article. We certainly **do not pretend** that they are the optimal models  across all the possible ones. The main purpose of sharing ASCAD is precisely to explore and evaluate new models.

We have performed our training sessions on two setups:
  * The first platform is composed of one gamer market **Nvidia GeForce GTX 1080 Ti**.
  * The second platform is composed of one professional computing market **Nvidia Tesla K80**.

Both setups were running an Ubuntu 16.04 distro with Keras 2.1.1 and TensorFlow-GPU 1.2.1. When using the GPU acceleration, the computation should not be very CPU and RAM intensive (at most one CPU core work load and 1 to 2 GB of RAM).
