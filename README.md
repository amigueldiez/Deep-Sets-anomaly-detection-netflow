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

- `deepsets-topk.py`:
  - Standalone script for training and evaluating the Deep Sets + top-k pooling model on the generated bags. It allow to remove the top-k pooling layer. Implements multiple random seeds and multiple runs for robust evaluation.

## Citations

```bibtex
@InProceedings{10.1007/978-3-032-29251-3_3,
author="Miguel-Diez, Alberto
and Campazas-Vega, Adri{\'a}n
and {\'A}lvarez-Aparicio, Claudia
and Sobr{\'i}n-Hidalgo, David
and Guerrero-Higueras, {\'A}ngel Manuel",
editor="Corchado, Emilio
and Quinti{\'a}n, H{\'e}ctor
and P{\'e}rez Garc{\'i}a, Hilde
and Calvo Rolle, Jos{\'e} Luis
and Ramos, S{\'e}rgio Filipe
and Mart{\'i}nez de Pis{\'o}n, Francisco Javier
and Fosci, Paolo",
title="Deep Sets for Network Flow Anomaly Detection Under a Multiple Instance Learning Framework",
booktitle="Computational Intelligence in Security for Information Systems",
year="2027",
publisher="Springer Nature Switzerland",
address="Cham",
pages="28--40",
abstract="The detection of malicious activity in high-throughput networks remains a challenging task due to the limitations of packet-level inspection and the growing volume of traffic generated in modern infrastructures. Flow-based analysis has emerged as a scalable alternative; however, most existing machine learning approaches operate at the individual flow level, which may fail to capture collective attack behaviors and often produce an excessive number of alerts. In this work, we propose a network intrusion detection framework based on Multiple Instance Learning (MIL), where network flows are grouped into bags and classified at the set level. To model the set-structured nature of the data, we employ a Deep Sets architecture enhanced with a top-k pooling mechanism, which allows the model to focus on the most informative instances within each bag and mitigates the dilution effect caused by predominantly benign traffic. The proposed approach is evaluated on the NF-CSE-CIC-IDS2018-v3 dataset, a widely used NetFlow-based benchmark. Experimental results demonstrate that the model achieves strong and balanced performance, obtaining an accuracy of 0.9528 and comparable precision and recall for both benign and malicious classes. These findings indicate that the proposed MIL-based Deep Sets framework is an effective and practical solution for flow-based network anomaly detection.",
isbn="978-3-032-29251-3"
}
```

## Acknowledgements

This research is a result of the CIBERLAB project (C083/23), carried out under the collaboration agreement between INCIBE and the University of León. This initiative is part of the Recovery, Transformation and Resilience Plan, funded by the European Union (Next Generation EU).

<img src="imgs/logo_proyecto.png" width="800" alt="Logo CIBERLAB" />

This research is also part of grant Explicit PID2024-162298OB-I00 funded by MICIU/AEI/ 10.13039/501100011033 and, as appropriate, by "ERDF A way of making Europe", by "ERDF/EU", by the "European Union".

<img src="imgs/Banda_de_logos_Explicit.png" width="500" alt="Logo EXPLICIT" />

Alberto Miguel-Diez was supported by an FPU fellowship from the _Ministerio de Ciencia, Innovación y Universidades_.

<img src="imgs/ministerio-universidades.png" width="400" alt="Logo Ministerio" />
