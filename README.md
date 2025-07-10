# RF_QRC
Recurrence-free quantum reservoir computing implementation along with standard classical and quantum reservoir approaches - to predict chaotic dynamics

This repository provides a framework for studying the stability of chaotic dynamical systems using **Quantum Reservoir Computing (QRC) and Recurrence-free Quantum reservoir computing (RF-QRC)**. We also use **Generalised Synchronisation (GS) theory** to design robust quantum reservoir computers. It includes tools for computing invariant properties such as **Lyapunov exponents**, **Covariant Lyapunov Vectors (CLVs)** and **Conditional Lyapunov Exponents (CLEs)** via a Qiskit-based implementation.


## Repository Structure

- **`src/Notebook.ipynb`**
  Contains the main workflow to evaluate stability properties with quantum reservoir computers. Input data can be modified to extend the framework to other systems such as **Lorenz96** for which the solvers are already added. Quantum noise models and the number of measurement **shots** can also be varied.

- **`src/QRC/qrc.py`**
  Implements the **Qiskit-based quantum reservoir** and the methods required to compute train in open-loop and predict in closed-loop using quantum circuit simulations.

  - **`src/QRC/crc.py`**
  Implements the classical reservoir computing or Echo State Network (ESN).

- **`src/QRC/solvers.py`**
  Classical solvers to generate the training / true data.

  - **`src/QRC/systems.py`**
  Implemented chaotic systems including Lorenz-63 , Lorenz-96 and MFE model.

  - **`src/QRC/validation.py`**
  Hyperparameter tuning (validation) using recycle validation to tune classical reservoir hyperparamters. RF-QRC does not require extensive hyperparamter tuning so the parameter can be directly varied in the notebook.


## Usage

To get started:
First, install the required dependencies:
```bash
pip install -r requirements.txt
```
Then run the following notebook
```bash

python Notebook.ipynb
```
Or alternatively,
```bash

python Notebook_MFE.ipynb
```

## Citation
If you use this code in your research, please cite the corresponding paper:

Prediction of chaotic dynamics and extreme events: A recurrence-free quantum reservoir computing approach (https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.043082)
