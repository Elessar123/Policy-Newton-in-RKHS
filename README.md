# Project Code Execution Guide 🚀

This document aims to explain how to run the supplementary code included in our project. This code is primarily divided into two parts: demo code for the **Toy Experiment** and experimental code run in the **Gym Environment**. The latter is programmed based on the **Cubic-regularized Policy Newton Algorithm (CRPN)**, and its original implementation can be found at [mizhaan23/crpn_algo](https://github.com/mizhaan23/crpn_algo).

---

## 📝 Code Structure Overview

Our supplementary code includes the following main parts:

1.  **Toy Experiment**: Used to demonstrate the basic operation of different algorithms in a simple environment.
2.  **Gym Environment Experiments**: Based on the OpenAI Gym environment, implementing four different reinforcement learning algorithms respectively.

---

## 🧪 Toy Experiment

For the **Toy Experiment**, we provide an integrated code file. This file will demonstrate the performance of four different algorithms in the same simple environment.


## 🏋️ Gym Environment Experiments

For the algorithm implementations in the **Gym Environment**, we have used four separate code files, each corresponding to a different algorithm. All these implementations are extended based on the framework or ideas of the [Cubic-regularized Policy Newton Algorithm (CRPN)](https://github.com/mizhaan23/crpn_algo).

The following are the individual code files and their corresponding algorithms:

1.  `reinforce.py`
    * **Algorithm**: Policy Gradient


2.  `reinforce_RKHS.py`
    * **Algorithm**: Policy Gradient in RKHS 


3.  `crpn.py`
    * **Algorithm**: Policy Newton


4.  `crpn_RKHS.py`
    * **Algorithm**: Policy Newton in RKHS 


---


