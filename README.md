# Physics-Informed Neural Networks for Solving Hyperbolic PDEs (XPINNs)

This repository implements an **Extended Physics-Informed Neural Network (XPINN)** to solve a system of **hyperbolic conservation laws** with discontinuous initial conditions.  

The approach partitions the computational domain into two subdomains to handle the **discontinuity at the interface** ($x = 0$), enabling more accurate learning of solutions across shock-like transitions.

---

## Governing Equations

The PDE system solved is:

$$
\begin{aligned}
\frac{\partial \rho}{\partial t} + \frac{\partial u}{\partial x} &= 0, \\
\frac{\partial u}{\partial t} + \frac{\partial \rho}{\partial x} &= 0,
\end{aligned}
$$

where $\rho(x,t)$ denotes density and $u(x,t)$ denotes velocity.

---

## Initial and Boundary Conditions

The initial conditions introduce a **discontinuity** at $x = 0$:

$$
\rho(x, 0) =
\begin{cases}
0.1, & x \le 0, \\
10,  & x > 0,
\end{cases}
\qquad
u(x, 0) =
\begin{cases}
2, & x \le 0, \\
1, & x > 0.
\end{cases}
$$

This sharp jump motivates the use of **XPINNs**, which divide the spatial domain into left and right subdomains:
- **Left subdomain ($x \le 0$):** modeled by `pinn_L`
- **Right subdomain ($x > 0$):** modeled by `pinn_R`  

The two are coupled through **interface continuity losses** that enforce matching $\rho$ and $u$ values at $x = 0$.

Zero-gradient boundary conditions are enforced at the domain edges ($x = \pm 1$):

$$
\frac{\partial \rho}{\partial x} = 0, \qquad \frac{\partial u}{\partial x} = 0
$$

---

## Network Architecture

Each subdomain is modeled by a fully connected neural network (MLP):

- 6 hidden layers  
- 64 neurons per layer  
- **tanh** activation after each layer

The **tanh** activator is smooth and continuously differentiable, making it well-suited for learning PDEs that require stable gradient computations.  
It also aids in approximating both linear and nonlinear solution features.

---

## Training Details

Training is handled via `train.py`, using **batched gradient descent** with the **Adam optimizer**.  

Batching allows for memory-efficient updates — each training iteration uses a subset of collocation, initial, boundary, and interface points instead of the entire dataset, stabilizing convergence and improving training throughput.

### **Total Loss Function**

The total loss is composed of:

$$
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{IC}} + 10 \cdot \mathcal{L}_{\text{interface}} + \mathcal{L}_{\text{BC}}
$$

where:
- $\mathcal{L}_{\text{PDE}}$ enforces the governing equations  
- $\mathcal{L}_{\text{IC}}$ enforces initial conditions  
- $\mathcal{L}_{\text{interface}}$ ensures continuity between subdomains  
- $\mathcal{L}_{\text{BC}}$ applies zero-gradient boundary conditions  

---

## Running the Code

To train and visualize results:

```bash
python main.py
```

This will:
1. Initialize two MLP models (`pinn_L` and `pinn_R`)  
2. Generate collocation, initial, interface, and boundary data  
3. Train the XPINN using batched optimization  
4. Save trained weights in `checkpoints/`  
5. Generate plots in `results/`

---

## Results

Below are placeholders for generated results — you can insert your actual images once you have them.

### Spacetime evolution:
![Spacetime plot](https://github.com/AbhinavVashishta/PINNS/blob/main/results/spacetime.png)

### Density $\rho(x,t)$ at fixed times:
- **t = 0.0:** ![rho_t0](https://github.com/AbhinavVashishta/PINNS/blob/main/results/solution_t00.png)  
- **t = 0.5:** ![rho_t05](https://github.com/AbhinavVashishta/PINNS/blob/main/results/solution_t05.png)  
- **t = 1.0:** ![rho_t10](https://github.com/AbhinavVashishta/PINNS/blob/main/results/solution_t10.png)

Each plot visualizes both subdomains and the smooth joining of $\rho$ and $u$ at the interface.

---

## Repository Structure

```
PINN_Project/
├── pinn/
│   ├── model.py         # MLP architecture with tanh activations
│   ├── physics.py       # PDE residual computation
│   ├── data.py          # Collocation and boundary data generation
│   ├── loss.py          # Combined loss with physics & interface terms
│   ├── train.py         # Training loop with batching
│   ├── utils.py         # Visualization and model saving
│   └── __init__.py
├── main.py              # Main entry point
├── results/             # Output plots
└── checkpoints/         # Trained model weights
```

---

## Key Insights

- **XPINNs** effectively handle discontinuities by splitting the domain and coupling sub-solutions via interface conditions.  
- **Batching** stabilizes training and allows for larger datasets.  
- **tanh activations** ensure differentiability for accurate PDE residuals.  
- This approach demonstrates how data-driven PDE solvers can capture sharp, physics-driven behaviors.

---

## References
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*  
- Jagtap, A. D., Kawaguchi, K., & Karniadakis, G. E. (2020). *Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition based Deep Learning Framework for Nonlinear PDEs.*

---

## Authors

This repository is a collaborative project under **Dr. Kirit Makwana's** Computational Physics course (EP4210) at IIT Hyderabad (Fall 2025).

Collaborators: [Abhinandan L](https://github.com/abhinandan2005)  [Abhinav Ganesh Vashishta](https://github.com/AbhinavVashishta)  [Samarth Gupta](https://github.com/samforarth)