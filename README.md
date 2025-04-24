# Quantum Dot Modeling Project

This project models electron confinement in quantum dots by solving the time-independent Schrödinger equation in one and two dimensions. The implementation uses both symbolic (SymPy) and numerical (SciPy) approaches to solve the differential equations governing quantum dot behavior.

## Features

- **1D Quantum Dot Modeling**
  - Square well potential
  - Harmonic potential
  - Gaussian potential
  - Interactive parameter adjustment
  - Real-time visualization

- **2D Quantum Dot Modeling**
  - Circular well potential
  - Animated time evolution
  - 3D surface visualization
  - Probability density plots

- **Visualization Tools**
  - Interactive dashboard with multiple plot types:
    * Wavefunction plots (real and imaginary parts)
    * Probability density plots
    * Energy level diagrams
    * Potential energy profiles
  - Animated 2D quantum dot evolution
  - Custom colormaps for quantum mechanical data
  - Real-time parameter adjustment

- **Technical Features**
  - Sparse matrix implementation for efficiency
  - Numerical eigenvalue solvers
  - Proper physical constants and units
  - Wavefunction normalization
  - Convergence analysis tools

## Requirements

- Python 3.8 or higher
- Dependencies listed in requirements.txt:
  - numpy>=1.21.0
  - scipy>=1.7.0
  - sympy>=1.9.0
  - matplotlib>=3.4.0
  - mpl_toolkits>=3.4.0
  - PyQt5 (for interactive visualization)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python quantum_dot_model.py
```

This will launch two interactive windows:
1. **Interactive Dashboard**: Shows 1D quantum dot visualizations with controls for:
   - Potential type selection (square, harmonic, gaussian)
   - Energy level adjustment
   - Quantum dot width adjustment
   - Real-time animation controls

2. **2D Quantum Dot Animation**: Displays the time evolution of a 2D quantum dot with:
   - 3D surface plot of probability density
   - Animated time evolution
   - Interactive play/pause controls

## Project Structure

- `quantum_dot_model.py`: Main implementation of quantum dot models
  - `QuantumDot1D` class for 1D modeling
  - `QuantumDot2D` class for 2D modeling
  - Energy level calculation functions
  - Main visualization setup

- `visualization.py`: Visualization module
  - Interactive dashboard creation
  - 2D animation tools
  - Custom plotting functions
  - Physical constants and utilities

- `requirements.txt`: Project dependencies
- `README.md`: Project documentation
- `LICENSE`: MIT License

## Theory

The project implements solutions to the time-independent Schrödinger equation:

\[ -\frac{\hbar^2}{2m}\nabla^2\psi + V(x)\psi = E\psi \]

where:
- \(\psi\) is the wavefunction
- \(V(x)\) is the potential energy
- \(E\) is the energy eigenvalue
- \(\hbar\) is the reduced Planck constant
- \(m\) is the particle mass

The implementation includes proper handling of physical constants:
- ħ (reduced Planck constant) = 1.0545718e-34 J·s
- m_e (electron mass) = 9.10938356e-31 kg
- eV (electron volt) = 1.602176634e-19 J
- e (elementary charge) = 1.602176634e-19 C
- ε₀ (vacuum permittivity) = 8.8541878128e-12 F/m

## License

This project is open source and available under the MIT License. See the LICENSE file for details. 