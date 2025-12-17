# Rationale

Major catastrophes often arise from the combination of cascading effects and amplifying environmental factors. Cascading effects unfold as chains-of-events, where each event has the potential to cause damage and trigger further consequences. The environment plays a critical role by shaping the conditions under which different perils can occur and interact. Multi-risk assessment therefore involves quantifying both the likelihood and the impact of such event chains.

Historical examples (see review by {cite:t}`Mignan_Wang2020`) illustrate this dynamic:
- Hurricane Katrina (2005) → storm surge → levee flood → lifeline failures → social unrest
- Tohoku earthquake (2011) → tsunami → nuclear accident → economic disruption
- COVID-19 pandemic → supply-chain failures → business interruption → global recession

The lack of sufficient data for statistical inference, combined with the vast range of interactions across natural, technological, and socio-economic systems, underscores the need for a deeper understanding of catastrophe dynamics within the complex Earth system. Indeed, the space of possible events is so large that historical catastrophes cover only a small fraction of what could occur. A framework like GenMR, grounded in physical principles and guided by reasoned imagination, enables the simulation of unforeseen super-catastrophes, capturing plausible extreme event chains that have not yet been observed.


## Compound events

A compound event is defined as an extreme event resulting from the interplay of multiple physical processes. It encompasses various forms of interconnected phenomena, including simple one-to-one chains of events (i.e., domino effects), external forcing factors simultaneously affecting multiple perils, combinations of events triggering subsequent events, feedback loops, and more complex scenarios that integrate all of these mechanisms {cite:p}`Seneviratne_etal2012,Mignan2024`. Compound events (Fig. 1) can be characterized by three main components:
1. Events: Represented as spheres in Figure 1, these include both individual loss-generating disasters, and "invisible" events {cite:p}`Mignan_etal2014`, that connect such disasters.
2. Stocks: In system dynamics terminology, stocks are environmental processes that act as sources of energy, driving or amplifying events. Examples include heat stocks contributing to warming or population stocks facilitating epidemic spread (depicted in Figure 1 as a thermometer and people, respectively).
3. Explicit interactions: Represented by arrows in Figure 1, these denote relationships between events and between events and environmental variables.

```{figure} figures/CompoundEvents.jpg
:width: 90%
:align: center

***Fig. 1. Concept of Compound Event.** (a) Combinations of causally unrelated events: Known as the "Perfect Storm" concept, this refers to simultaneous events in the same region, such as an earthquake and a storm. (b) Simple one-to-one chains of events: Examples include successive explosions, fires, and toxic spills at critical infrastructures. (c) Common external forcing factor: A shared external influence, such as warming, alters the probability or intensity of multiple events—for instance, droughts, heatwaves, and wildfires. (d) System feedback loops: Feedback mechanisms that reinforce events, such as the escalation of an epidemic into a pandemic. Compound events may also involve intricate combinations of all the above factors.*
```


## Generic Multi-Risk (GenMR) framework

The Generic Multi-Risk (GenMR) framework {cite:p}`Mignan_etal2014` is a probabilistic multi-risk approach that generates chains-of-events using a Monte Carlo simulation procedure, with an adjacency matrix of event interactions at its core. The framework naturally accounts for random co-occurrences of events, while also incorporating the memory of previous states, which can further amplify risk—for example, via dynamic vulnerability {cite:p}`Mignan_etal2018`. Additionally, GenMR can model non-stationary processes, such as seasonal variations, that influence the likelihood and impact of events {cite:p}`Matos_etal2015`.

GenMR follows the established standards of catastrophe (CAT) risk modelling. It employs the hazard → risk workflow, with the event loss table (ELT) and year-loss table (YLT) as primary output formats, and uses common risk metrics such as average annual loss (AAL) and exceedance probability (EP) curves {cite:p}`Mignan2024`.

The main innovation of GenMR lies in its interaction-centric approach to probabilistic risk modelling (Fig. 2) and its generic format. The former simplifies the treatment of dynamic processes in multi-risk assessment, while the latter enables the modelling of any combination of perils by categorizing and harmonizing them according to the simplest possible rules {cite:p}`Mignan2022a`. This is complemented by the concept of the digital template, presented in the next section.

```{figure} figures/GenMRcore_adjmatrix.jpg
:width: 100%
:align: center

***Fig. 2. Event interaction modelling in GenMR.** Chains-of-events are implemented in a Year Event Table (YET) using a transition matrix of conditional probabilities. Adding event losses produces the Year Loss Table (YLT). Reproduced from {cite:t}`Mignan2024`.*
```

### Past developments

GenMR was conceived by <a href="https://www.linkedin.com/in/amignan/" target="_blank">A. Mignan</a> within the context of _New Multi-HAzard and MulTi-RIsK Assessment MethodS for Europe_ (<a href="https://cordis.europa.eu/project/id/265138" target="_blank">MATRIX 2010-2013</a>, grant 265138) {cite:p}`Mignan_etal2014,Komendantova_etal2014,Scolobig_etal2017,Mignan_etal2017` and was further developed for the _Harmonized approach to stress tests for critical infrastructures against natural hazards_ (<a href="http://www.strest-eu.org/" target="_blank">STREST 2013-2016</a>, grant 603389) {cite:p}`Mignan_etal2018,Matos_etal2015`. Both projects were funded by the  European Union’s Seventh Framework Programme, with GenMR developed at the Swiss Federal Institute of Technology Zurich (ETH Zurich). Work later resumed at the Institute of Risk Analysis, Prediction and Management (Risks-X) at the Southern University of Science and Technology (SUSTech), supported by funding from the _National Natural Science Foundation of China_ (2021-2022, grant 42074054) {cite:p}`Mignan2022b`.

Several approaches have been developed to quantify one-to-one interactions in the GenMR adjacency matrix, including: expert elicitation and reasoned imagination conducted through stakeholder workshops and other tools {cite:p}`Komendantova_etal2014,Mignan_etal2016,Mignan_etal2022b`, empirical analysis of dozens of past super-catastrophes {cite:p}`Mignan_Wang2020`, and frequentist encoding of chains of events at oil and gas infrastructures using the ENSAD database of severe energy accidents {cite:p}`Mignan_etal2022a`. 

### New developments

To date, no dedicated GenMR platform has been developed. Instead, various early and incomplete prototypes were created for the projects mentioned above, serving as preliminary demonstrations of the framework’s potential. Beginning in Fall 2025, GenMR is being formally developed by Mignan Risk Analytics GmbH as a Python package, supported by funding from the <a href="https://foundation.scor.com/funded-projects/generic-multi-risk-genmr-open-source-platform-model-compound-catastrophes" target="_blank">SCOR Foundation for Science</a>.


```{warning}
This package is under active development. Version 1.1.1 has been released in December 2025. Version 1.1.2 is scheduled for release by the end of June 2026 — please check back then for the update.
```



## Digital Template

Multi-risk assessment is typically explored either from a theoretical perspective or through localized applications. The former often remains abstract, while the latter is constrained by site-specific conditions. The GenMR digital template fills the gap between these two extremes by providing a sandbox for multi-risk R&D and for exploring a wide range of plausible scenarios. Formalised by {cite:t}`Mignan2022b` and first applied in the <a href="https://github.com/amignan/Intro2CATriskModelling/blob/main/CATRiskModellingSandbox_tutorial.ipynb" target="_blank">CAT Risk Modelling Sandbox</a> {cite:p}`Mignan2024`, the digital template is particularly suited for GenMR prototyping and dynamic-risk modelling education. It can be thought of as a microcosm simulation of the complex Earth system: a virtual environment populated by loss-generating events that interact with one another and with their environment, with dynamics and risk outputs simulated by GenMR. The term, digital template, refers to the generic blueprint required as a foundation for any future advanced digital twin multi-risk application.


### Past developments

The default setting was sketched by <a href="https://www.linkedin.com/in/amignan/" target="_blank">A. Mignan</a> within the scope of the MATRIX project, where the digital template was originally coined Virtual City {cite:p}`Komendantova_etal2014,Liu_etal2015,Mignan_etal2017` (Fig. 3). After remaining dormant for several years, it was revived at Risks-X in 2020–2022. During this period, the digital template was formalised and expanded through the systematic collection of hazard and interaction models for more than 20 perils, described in the IJERPH trilogy {cite:p}`Mignan_Wang2020`-{cite:p}`Mignan2022a`-{cite:p}`Mignan2022b`. In the <a href="https://github.com/amignan/Intro2CATriskModelling/blob/main/CATRiskModellingSandbox_tutorial.ipynb" target="_blank">CAT Risk Modelling Sandbox</a>, the environmental layers of the digital template (five in total, see below) are implemented as a series of .npy arrays. While this design choice kept the framework lightweight—appropriate for a textbook introducing catastrophe modelling concepts {cite:p}`Mignan2024`—it limited the possibility of alternative parameterizations.

```{figure} figures/VirtualRegion_sketch.jpg
:width: 100%
:align: center

***Fig. 3. The virtual environment envisioned for the Virtual City concept** {cite:p}`Komendantova_etal2014,Liu_etal2015,Mignan_etal2017`. Its digitalization, structured into multiple environmental layers, was realized with the development of the digital template {cite:p}`Mignan2022b`. Illustrations by M. Deichmann.*
```

### New developments

The digital template is central to multi-risk R&D and GenMR prototyping. Every environment (`GenMR.environment.py`), peril (`GenMR.perils.py`), and associated dynamic process (`GenMR.dynamics.py`) will be implemented within this template, which is designed to be fully parameterizable by the user. Unlike the CAT Risk Modelling Sandbox where processes were implemented as standalone Python functions (and multi-risk capabilities were absent), the GenMR package adopts an object-oriented design, with most functionalities encapsulated in Python classes.


## Tutorials

One can explore the possibilities offered by the GenMR package through the following tutorials:
- {doc}`notebooks/Tutorial1_DigitalTemplate_environment`: A step-by-step guide to building the digital template environment.
    - Version 1.1.1 includes five environmental layers: topography, soil, land use, urban surface, and road network.
- {doc}`notebooks/Tutorial2_DigitalTemplate_perils`: A description of how perils can be implemented in the  digital template environment, as well as a tutorial on CAT risk modelling, covering the creation of an event loss table (ELT), event footprint catalogue, hazard maps and hazard curves, and exceedance probability (OEP) curves, applied to the previously virtual built environment.
    - Version 1.1.1 features ten perils: asteroid impacts (AI), earthquakes (EQ), fluvial floods (FF), landslides (LS), rainstorms (RS), storm surges (SS), tropical cyclones (TC), volcanic eruptions (VE), wildfires (WF), and industrial explosions (Ex).
- Tutorial 3 COMING IN 2026: A tutorial on catastrophe dynamics, demonstrating how to model chains-of-events and their cascading impacts.



## How-to guides

How-to guides will be released toward the end of the SCOR project (Fall 2027). These guides will demonstrate how to address specific multi-risk problems using the digital template.



## References

```{bibliography}
:style: unsrt
```