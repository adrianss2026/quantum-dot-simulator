# Quantum Dot Modeling Project

This project models electron confinement in quantum dots by solving the time-independent Schrödinger equation in one and two dimensions. The implementation uses both symbolic (SymPy) and numerical (SciPy) approaches to solve the differential equations governing quantum dot behavior.

## Features

- One-dimensional quantum dot modeling
- Two-dimensional quantum dot modeling
- Analytical solutions using SymPy
- Numerical solutions using SciPy's solve_ivp
- Visualization of wavefunctions and probability densities
- Comparison of analytical and numerical solutions

## Requirements

- Python 3.8 or higher
- Dependencies listed in requirements.txt

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

## Project Structure

- `quantum_dot_model.py`: Main script containing the quantum dot modeling implementation
- `visualization.py`: Module for plotting wavefunctions and probability densities
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Theory

The project implements solutions to the time-independent Schrödinger equation:

\[ -\frac{\hbar^2}{2m}\nabla^2\psi + V(x)\psi = E\psi \]

where:
- \(\psi\) is the wavefunction
- \(V(x)\) is the potential energy
- \(E\) is the energy eigenvalue
- \(\hbar\) is the reduced Planck constant
- \(m\) is the particle mass

## License

This project is open source and available under the MIT License. 