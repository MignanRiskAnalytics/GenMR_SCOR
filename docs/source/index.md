% GenMR documentation master file

# GenMR

GenMR is an open-source Python package for research and development in catastrophe dynamics. It provides the probabilistic Generic Multi-Risk (GenMR) framework along with a digital template for multi-risk simulations in a fully parameterizable virtual environment. The catastrophe dynamics process is structured into three sequential steps:
1. Virtual environment generation - handled by `GenMR.environment.py`
2. Definition of stochastic events for individual perils - handled by `GenMR.perils.py` (*upcoming*)
3. Simulation of chains-of-events and risk drivers - handled by `GenMR.dynamics.py` (*planned for Fall 2026*).

Other modules are also planned:
- `GenMR.utils.py` - general-purpose utilities
- `GenMR.assess.py` - risk assessment tools

---

This work is being developed as part of the 2025–2027 project "A Generic Multi-Risk (GenMR) Open-Source Platform to Model Compound Catastrophes", funded by the <a href="https://foundation.scor.com/funded-projects/generic-multi-risk-genmr-open-source-platform-model-compound-catastrophes" target="_blank">SCOR Foundation for Science</a>.

```{image} figures/SCOR_Foundation_for_Science_Logo_Blue.png
:width: 50%
:align: center
:target: https://foundation.scor.com
```

The primary objective of this project is to advance our understanding of compound events through the development of the GenMR platform. By compound events, we refer to domino effects, cascading processes, and other amplification mechanisms that can escalate into super-catastrophes. GenMR is designed to help users address fundamental research and risk management questions, such as:
- What are the dominant patterns of event chains that drive large losses?
- Which critical events or drivers contribute most to the buildup of systemic risk?
- How does the generic risk profile — measured through indicators such as annual average loss (AAL), exceedance probability (EP) curves, and related metrics — change under different model parameterizations?

The GenMR framework and digital template are designed to be generic by construction. To capture the fundamentals of multi-risk, they employ the simplest possible models, offering a parsimonious yet effective representation of complex Earth system dynamics. At the same time, GenMR aims to be comprehensive in scope, enabling the exploration of potential interactions both within and across natural, technological, and socio-economic systems.

```{figure} figures/digitaltemplate_env_rayshader.jpg
:width: 90%
:align: center

*Example of a virtual environmental in the GenMR digital template - Rendered with <a href = "https://www.rayshader.com/" target = "_blank">Rayshader</a> (by T. Morgan-Wall). In version 1.1.1, the digital template is identical to that developed for the <a href = "https://github.com/amignan/Intro2CATriskModelling/blob/main/CATRiskModellingSandbox_tutorial.ipynb" target = "_blank">CAT Risk Modelling Sandbox</a>, part of the Cambridge University Press textbook <a href="https://www.cambridge.org/highereducation/books/introduction-to-catastrophe-risk-modelling/A3A5B5FB990921422BFEBB07734BF869#overview" target = "_blank">Introduction to Catastrophe Risk Modelling - A Physics-based Approach</a> (Mignan, 2024).*
```


**GenMR_SCOR Project Timeline**
1. *GenMR blueprint preparation* (v1.1.1, due Dec. 2025): Enhancement of the <a href = "https://github.com/amignan/Intro2CATriskModelling/blob/main/CATRiskModellingSandbox_tutorial.ipynb" target = "_blank">CAT Risk Modelling Sandbox</a> from Mignan (2024), Cambridge University Press, by transitioning to an object-oriented Python platform.
2. *Implementation of additional perils & environmental layers* (v1.1.2, due Jun. 2026): Development of climatic and socio-economic perils, which were not included in the 2024 CAT Risk Modelling Sandbox.
3. *Implementation of Multi-Risk Core* (v1.2.1, due Sep. 2026): Modelling of chains-of-events and risk drivers within the GenMR framework.
4. *Implementation of hydropower-dam and dam flood model* (led by UniL project partner, v1.2.2, due Mar. 2027): Reproduction of Matos et al. (2015;2017) and integration into digital template.
5. *Interaction matrix encoding* (v1.2.3, due Jun. 2027): Quantitative assessment of event–event and event–environment interactions within the digital template.

<br>

```{warning}
This package is under active development. Version 1.1.1 is scheduled for release by the end of December 2025 — please check back then for the update.
```

To install GenMR:

```console
COMING LATER
```


```{toctree}
:caption: 'Contents:'
:maxdepth: 2

rationale
manual
Tutorial 1: Environment generation<notebooks/Tutorial1_DigitalTemplate_environment>
Tutorial 2: Peril implementation (in construction)<notebooks/Tutorial2_DigitalTemplate_perils>
```

