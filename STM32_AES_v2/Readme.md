## The ASCADv2 database

ANSSI has provided source code implementation of a protected AES-128 encryption and decryption for 32-bit Cortex-M ARM, which can be found on the following github repository: [ANSSI-FR/secAES-STM32](https://github.com/ANSSI-FR/SecAESSTM32). This implementation is more secure than the [ATMega8515 implementation used for ASCADv1](https://github.com/ANSSI-FR/secAES-ATmega8515), since it combines two side channel countermeasures: affine masking and shuffling. Details of the implementation may be read in the second part of the Readme file in [ANSSI-FR/secAES-STM32/](https://github.com/ANSSI-FR/SecAESSTM32) and a security analysis is detailed in the dedicated report [doc/technical-report/technical_analysis.pdf](https://github.com/ANSSI-FR/SecAESSTM32/blob/master/doc/technical-report/technical_analysis.pdf). To sum-up, this implementation is more secure than the [ATMega8515 implementation used for ASCADv1](https://github.com/ANSSI-FR/secAES-ATmega8515).  To test it, we measured the power consumption of a STM32 Cortex M4 microcrontroller (STM32F303RCT7) during 800,000 AES encryptions with random keys and random plaintexts. Traces have been acquired with a ChipWhisperer board [CW308T-STM32F](https://wiki.newae.com/CW308T-STM32F) by underclocking the STM32 clock to 4 MHz and acquired though an oscilloscope with a 50,000,000 samples per second rate. The measured traces consist in 1,000,000 samples points, encompassing the whole AES encryption. The resulting dataset is publicly available on the data.gouv.fr platform: [ASCADv2](https://www.data.gouv.fr/en/datasets/ascadv2/).

## <a name="getting-ascadv2"> Getting the ASCADv2 databases and the trained models 

In the new folder, first download the data packages with the raw data by using:

<pre>
$ cd STM32_AES_v2
$ mkdir -p ASCAD_data/ASCAD_databases
$ cd ASCAD_data/ASCAD_databases
$ wget https://www.data.gouv.fr/api/1/datasets/r/a6cf925c-079c-4468-a723-d94bce6c31f8 -O ascadv2-extracted.h5
$ wget https://www.data.gouv.fr/api/1/datasets/r/78697eec-7adf-4f68-bffa-b55640b40478 -O ascadv2-stm32-conso-raw-traces1.h5
$ wget https://www.data.gouv.fr/api/1/datasets/r/b3a41a00-3552-40a6-a18e-d7f1b97596ae -O ascadv2-stm32-conso-raw-traces2.h5
$ wget https://www.data.gouv.fr/api/1/datasets/r/b3775bf7-8657-4b02-964c-7cafa3ff8d06 -O ascadv2-stm32-conso-raw-traces3.h5
$ wget https://www.data.gouv.fr/api/1/datasets/r/4f489a9c-70d8-487f-ad95-5ce2a6af9cdd -O ascadv2-stm32-conso-raw-traces4.h5
$ wget https://www.data.gouv.fr/api/1/datasets/r/98988df2-80ec-4f0a-af8f-32b38ccd742e -O ascadv2-stm32-conso-raw-traces5.h5
$ wget https://www.data.gouv.fr/api/1/datasets/r/651a3ba2-bced-4f3b-8287-a40935b4e60f -O ascadv2-stm32-conso-raw-traces6.h5
$ wget https://www.data.gouv.fr/api/1/datasets/r/8f32464b-6bdf-4720-9890-b7048e9822f1 -O ascadv2-stm32-conso-raw-traces7.h5
$ wget https://www.data.gouv.fr/api/1/datasets/r/bea66e8a-75c2-43c3-ae38-f4dc26ae2df5 -O ascadv2-stm32-conso-raw-traces8.h5
</pre>

Please be aware that all these steps should **download around 807 GB** of data.
You can selectively download only the [extracted database](https://files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5) that weights "only" 7 GB:


<pre>
$ wget https://www.data.gouv.fr/api/1/datasets/r/a6cf925c-079c-4468-a723-d94bce6c31f8 -O ascadv2-extracted.h5
</pre>


Once this step is over, you can download the trained models:

<pre>
$ mkdir ../ASCAD_trained_models
$ cd ../ASCAD_trained_models
$ wget https://www.data.gouv.fr/api/1/datasets/r/a4eeef49-2c6a-40e3-9e8e-f8ecccb52974 -O ascadv2-multi-resnet-earlystopping.zip
$ wget https://www.data.gouv.fr/api/1/datasets/r/c26d7e7c-f1ff-4ba0-9384-983bf62d00e4 -O ascadv2-multi-resnet-wo-permind-earlystopping.zip
$ unzip ascadv2-multi-resnet-earlystopping.zip
$ unzip ascadv2-multi-resnet-wo-permind-earlystopping.zip
</pre>
The first model corresponds to a multitask ResNet model that predicts all the intermediate mask values (multiplicative mask, additive mask, shuffled masked sbox output for each index i in [1..16], shuffle index for i in [1..16]). The second model ignores that the implementation has shuffled the sbox outputs and classifies directly the multiplicative mask, additive mask, and the masked sbox outputs without shuffle. More details about the methodology and the two models may be found in our [eprint article](https://eprint.iacr.org/2021/592).

## Test the trained models

<pre>
$ cd ../../..
$ python ASCAD_test_models.py ./STM32_AES_v2/example_test_models_params # if you want to test the first trained model
$ python ASCAD_test_models.py ./STM32_AES_v2/example_test_models_without_permind_params # if you want to test the second trained model
</pre>

The script performs a recombination of the labels probabilities and computes a Maximum Likelyhood Estimation (MLE) for each possible value of the key bytes. Then the rank of the correct key (obtained from the MLE order) is displayed for each byte of the key. The script validates the success of our attack since all the correct key byte values are ranked first after approximatively 120 test traces for the first trained model, and 250 test traces for the second model. 

## Perform the attack from the raw dataset
If you want to test all the steps of the attack from scratch, you can extract the 15,000 points of interest from the raw dataset by using:

<pre>
$ python ASCAD_generate.py ./STM32_AES_v2/example_generate_params
</pre>

Then you can train each of the models by using:

<pre>
$ python ASCAD_train_models.py ./STM32_AES_v2/example_train_models_params
$ python ASCAD_train_models.py ./STM32_AES_v2/example_train_models_without_permind_params
</pre>

## Raw data files hashes

The data files SHA-1 hash values are available [here](https://files.data.gouv.fr/anssi/ascadv2/sha1.txt).

