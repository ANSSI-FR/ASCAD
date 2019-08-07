## The ATMega8515 SCA traces databases

ANSSI has provided source code implementations of two **masked AES** on the ATMega8515 MCU target, which can be found on the following github repository: [ANSSI-FR/secAES-ATmega8515](https://github.com/ANSSI-FR/secAES-ATmega8515). ATMega8515 uses external clocking: the acquisitions have been performed using a smart card reader providing a **4 MHz clock**, compatible with the ISO7816-3 standard default values.

The current ASCAD databases correspond to the first version (v1) of the masked AES (the second version has improved security and is currently kept for further studies). The v1 implementation allowed us to perform the acquisition of Power consumptions measurements in traces of 100,000 time samples (at 2 giga-samples per second) on an [ATMega8515](http://www.infinityusb.com/default.asp?show=store&ProductGrp=8) based [WB Electronics 64 Kbit ATMega chipcard](http://www.infinityusb.com/default.asp?show=store&ProductGrp=8) (see the [secAES-ATmega8515](https://github.com/ANSSI-FR/secAES-ATmega8515) material for more information). No specific hardware countermeasure has been activated on the ATMega8515.

Two campaigns are available in the current sub-folders. The first campaign corresponds to a setting where the **key is fixed** for all measurements, and the second campaign corresponds to a setting where the **key is variable** for 66% of the measurements (it is fixed for the remaining 33% which allows for the building of an attack/testing database). More information can be found in the sub-folders.

Instructions to download the databases are available in the corresponding sub-folders:

* **Fixed** key version: [ATMEGA_V1 fixed](./ATM_AES_v1_fixed_key)
* **Variable** key version: [ATMEGA_V1 variable](./ATM_AES_v1_variable_key)
