# Deep Sets for Network Flow Anomaly Detection (MIL)

This repository contains the reference implementation (Jupyter notebook) used in the paper **“Deep Sets for Network Flow Anomaly Detection under a Multiple Instance Learning Framework”**.

The core idea is to model bags of NetFlow records under the Multiple Instance Learning (MIL) paradigm and classify each bag as *benign* or *malicious* using a Deep Sets architecture enhanced with top-k pooling.



## Contents

- `deepsets_def.ipynb`:
  - Loads the NetFlow dataset (`NF-CICIDS2018-v3.csv`)
  - Basic EDA and dataset inspection
  - Preprocessing utilities (feature filtering, NaN/Inf cleaning)
  - Feature importance / feature selection 
  - Bag generation for MIL
  - Deep Sets + top-k pooling training and evaluation

## Acknowledgements

This research is a result of the CIBERLAB project (C083/23), carried out under the collaboration agreement between INCIBE and the University of León. This initiative is part of the Recovery, Transformation and Resilience Plan, funded by the European Union (Next Generation EU).

<img src="imgs/logo_proyecto.png" width="300" alt="Logo CIBERLAB" />

This research is also part of grant Explicit PID2024-162298OB-I00 funded by MICIU/AEI/ 10.13039/501100011033 and, as appropriate, by "ERDF A way of making Europe", by "ERDF/EU", by the "European Union".

<img src="imgs/Banda_de_logos_Explicit.png" width="300" alt="Logo EXPLICIT" />

Alberto Miguel-Diez was supported by an FPU fellowship from the _Ministerio de Ciencia, Innovación y Universidades_.

<img src="imgs/ministerio-universidades.png" width="300" alt="Logo Ministerio" />
