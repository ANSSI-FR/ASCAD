## The ASCADv1 database

This sub-repository is associated with the article ["Study of Deep Learning Techniques for
Side-Channel Analysis and Introduction to ASCAD Database"](https://eprint.iacr.org/2018/053.pdf) available on the [eprints](https://eprint.iacr.org).

Databases, Neural Networks models as well as scripts are provided here as a complementary
material to the article: please refer to it for various explanations and details about SCA and Deep Learning.

## Authors

  * Ryad BENADJILA (<mailto:ryad.benadjila@ssi.gouv.fr>)
  * Eleonora CAGLI (<mailto:eleonora.cagli@cea.fr>)
  * Cécile DUMAS (<mailto:cecile.dumas@cea.fr>)
  * Emmanuel PROUFF (<mailto:emmanuel.prouff@ssi.gouv.fr>)
  * Rémi STRULLU (<mailto:remi.strullu@ssi.gouv.fr>)
  * Adrian THILLARD (<mailto:adrian.thillard@ssi.gouv.fr>)

## Acknowledgements

This work has been partially funded through the H2020 project REASSURE.

## The ATMega8515 SCA campaigns

ANSSI has provided source code implementations of two **masked AES** on the ATMega8515 MCU target, which can be found on the following github repository: [ANSSI-FR/secAES-ATmega8515](https://github.com/ANSSI-FR/secAES-ATmega8515). ATMega8515 uses external clocking: the acquisitions have been performed using a smart card reader providing a **4 MHz clock**, compatible with the ISO7816-3 standard default values.

These ASCAD databases correspond to the first version (v1) of the masked AES (the second version has improved security and is available [here](../STM32_AES_v2)). The v1 implementation allowed us to perform the acquisition of Power consumptions measurements in traces of 100,000 time samples (at 2 giga-samples per second) on an [ATMega8515](http://www.infinityusb.com/default.asp?show=store&ProductGrp=8) based [WB Electronics 64 Kbit ATMega chipcard](http://www.infinityusb.com/default.asp?show=store&ProductGrp=8) (see the [secAES-ATmega8515](https://github.com/ANSSI-FR/secAES-ATmega8515) material for more information). No specific hardware countermeasure has been activated on the ATMega8515.

Two campaigns are available in the current sub-folders. The first campaign corresponds to a setting where the **key is fixed** for all measurements, and the second campaign corresponds to a setting where the **key is variable** for 66% of the measurements (it is fixed for the remaining 33% which allows for the building of an attack/testing database). More information can be found in the sub-folders.

Instructions to download the databases are available in the corresponding sub-folders:

* **Fixed** key version: [ATMEGA_V1 fixed](./ATM_AES_v1_fixed_key)
* **Variable** key version: [ATMEGA_V1 variable](./ATM_AES_v1_variable_key)
