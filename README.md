# [The Rumble in the Millimeter Wave Jungle: Obstructions Vs RIS](https://ieeexplore.ieee.org/document/10578262)

[Juan Marcos Ramirez](https://juanmarcosramirez.github.io/), [Vincenzo Mancuso](https://networks.imdea.org/es/team/imdea-networks-team/people/vincenzo-mancuso/), and [Marco Ajmone Marsan](https://networks.imdea.org/es/team/imdea-networks-team/people/marco-ajmone-marsan/)

 22st Mediterranean Communication and Computer Networking Conference (MedComNet 2024)

## Abstract

Reconfigurable intelligent surfaces (RISs) have emerged as a key technology for future communication systems. RISs are arrays of tunable reflecting elements that provide controllable propagation channels by smartly shaping incident electromagnetic (EM) waves. Analysis and improvement of RIS-aided systems require the definition of accurate path loss models that consider environmental effects often encountered in practical applications. In this paper, we derive a path loss model for RIS-assisted communications to account for the attenuation induced by the transmission medium and randomly located obstructions. More precisely, this study focuses on assessing the impact caused by Poisson-located obstructing objects on RIS-assisted millimeter wave links. To this end, we evaluate the outage probability yielded by RIS-aided systems in indoor environments with antenna beam-steering and random obstructions. We obtain extensive simulation results to assess the impact of RIS considering different parameters, such as the minimum signal-to-noise ratio (SNR) necessary for successful reception, the operating frequency, the density of the Poisson process used for object placement, and the object size.

## Bibtex

```
@inproceedings{ramirez2024rumble,
  author={Ramírez, Juan Marcos and Mancuso, Vincenzo and Ajmone Marsan, Marco},
  booktitle={2024 22nd Mediterranean Communication and Computer Networking Conference (MedComNet)}, 
  title={The Rumble in the Millimeter Wave Jungle: Obstructions Vs Ris}, 
  year={2024},
  pages={1-10},
  doi={10.1109/MedComNet62012.2024.10578262}}
```

![Communication System](https://github.com/JuanMarcosRamirez/outage_RIS_indoor/blob/main/images/RIS_assisted_system.jpg?raw=true "Demo image")

## How to run the code

The performance evaluation of RIS-assisted wireless communication systems in the presence of randomly positioned obstructions was executed interactively using Jupyter notebooks across different scenarios. 

### 2D Indoor Case

#### Figure 2

* To generate the curves in Figure 2(b), run `Poutage_vs_gamma.ipynb` file in the indoor2D directory. Adjust the operating frequency and the number of RIS elements to obtain different curves.

* To generate the curves in Figure 2(c), run `Poutage_vs_lambda_ppp.ipynb` file in the indoor2D directory. Adjust the operating frequency and the number of RIS elements to obtain different curves.

* To generate the curves in Figure 2(d), run `Poutage_vs_radius.ipynb` file in the indoor2D directory. Adjust the operating frequency and the number of RIS elements to obtain different curves.

#### Figure 4

* To generate the curves in Figure 4(a), run `Without_Beamsteering.ipynb` file in the indoor2D directory. Adjust the operating frequency and the number of RIS elements to obtain different curves.

### 3D Indoor Case

#### Figure 5

* To generate the curves in Figure 5, run `Poutage_vs_gamma_3D.ipynb` file in the indoor3D directory. Adjust the operating frequency and the number of RIS elements to obtain different curves.

## Platform

The code has been executed in a Linux environment, specifically on the Ubuntu 22 Operating System.

## Acknowledgments

This work has been supported by Project AEON-CPS (TSI-063000-2021-38), funded by the Ministry of Economic Affairs and Digital Transformation and the European Union NextGeneration-EU in the Spanish Recovery, Transformation, and Resilience Plan framework.

## License

This code package is licensed under the GNU GENERAL PUBLIC LICENSE (version 3) - see the [LICENSE](LICENSE) file for details


## Author

Juan Marcos Ramírez Rondón. Postdoctoral Researcher. [IMDEA Networks Institute](https://networks.imdea.org/es/). Leganés, 28918, Spain. 


### Contact

[Juan Marcos Ramirez](juan.ramirez@imdea.org)

## Date

June 16th, 2024
