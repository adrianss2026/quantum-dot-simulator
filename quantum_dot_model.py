import numpy as np
from scipy.sparse import kron, diags
from scipy.sparse.linalg import eigsh
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from visualization import (plot_wavefunction, plot_probability_density, plot_2d_wavefunction,
                         animate_wavefunction, plot_energy_levels, create_interactive_dashboard,
                         plot_animated_2d_evolution, hbar, m_e, eV, e, epsilon_0)
from typing import Tuple, List, Callable, Union

class QuantumDot1D:
    """1D Quantum Dot model with various potential types."""
    
    def __init__(self, width=10e-9, depth=0.3*1.602176634e-19, n_points=1000, potential_type='square'):
        """
        Initialize 1D Quantum Dot.
        
        Parameters:
        -----------
        width : float
            Width of the quantum dot in meters
        depth : float
            Depth of the potential well in Joules
        n_points : int
            Number of grid points
        potential_type : str
            Type of potential ('square', 'harmonic', or 'gaussian')
        """
        self.width = width
        self.depth = depth
        self.n_points = n_points
        self.potential_type = potential_type
        
        # Set up spatial grid
        self.x = np.linspace(-2*width, 2*width, n_points)
        self.dx = self.x[1] - self.x[0]
        
        # Create potential
        self.V = self._create_potential()
        
        # Create Hamiltonian
        self.H = self._create_hamiltonian()
    
    def _create_potential(self):
        """Create the potential energy function."""
        if self.potential_type == 'square':
            V = np.zeros_like(self.x)
            mask = np.abs(self.x) <= self.width/2
            V[mask] = -self.depth
        elif self.potential_type == 'harmonic':
            omega = np.sqrt(2 * self.depth / (self.width**2 * m_e))
            V = 0.5 * m_e * omega**2 * self.x**2 - self.depth
        elif self.potential_type == 'gaussian':
            V = -self.depth * np.exp(-2 * (self.x/self.width)**2)
        else:
            raise ValueError(f"Unknown potential type: {self.potential_type}")
        return V
    
    def _create_hamiltonian(self):
        """Create the Hamiltonian matrix using sparse matrices."""
        N = self.n_points
        dx = self.dx
        
        # Kinetic energy term (second derivative)
        diagonals = [1, -2, 1]
        positions = [-1, 0, 1]
        T = diags(diagonals, positions, shape=(N, N)) * (-hbar**2 / (2 * m_e * dx**2))
        
        # Potential energy term
        V_matrix = diags(self.V, 0)
        
        return T + V_matrix
    
    def numerical_solution(self, n_states=1):
        """
        Solve for eigenstates numerically.
        
        Parameters:
        -----------
        n_states : int
            Number of eigenstates to compute (must be positive)
            
        Returns:
        --------
        energies : ndarray
            Array of energy eigenvalues
        wavefunctions : ndarray
            Array of wavefunction values
        """
        # Ensure n_states is positive
        n_states = max(1, int(n_states))
        
        # Solve for eigenstates
        energies, states = eigsh(self.H, k=n_states, which='SA')
        
        # Normalize wavefunctions
        for i in range(n_states):
            norm = simpson(y=np.abs(states[:, i])**2, x=self.x)
            states[:, i] /= np.sqrt(norm)
        
        return energies, states

class QuantumDot2D:
    """2D Quantum Dot model."""
    
    def __init__(self, width=10e-9, depth=0.3*1.602176634e-19, n_points=50):
        """
        Initialize 2D Quantum Dot.
        
        Parameters:
        -----------
        width : float
            Width of the quantum dot in meters
        depth : float
            Depth of the potential well in Joules
        n_points : int
            Number of grid points in each dimension
        """
        self.width = width
        self.depth = depth
        self.n_points = n_points
        
        # Set up spatial grid
        self.x = np.linspace(-2*width, 2*width, n_points)
        self.y = np.linspace(-2*width, 2*width, n_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        
        # Create potential
        self.V = self._create_potential()
    
    def _create_potential(self):
        """Create 2D potential energy function (circular well)."""
        R = np.sqrt(self.X**2 + self.Y**2)
        V = np.zeros_like(R)
        mask = R <= self.width/2
        V[mask] = -self.depth
        return V.flatten()
    
    def solve_2d(self, n_eigenstates=1, sparse=True):
        """
        Solve the 2D Schrödinger equation.
        
        Parameters:
        -----------
        n_eigenstates : int
            Number of eigenstates to compute
        sparse : bool
            Whether to use sparse matrices (recommended for large systems)
            
        Returns:
        --------
        energies : ndarray
            Array of energy eigenvalues
        wavefunctions : ndarray
            Array of wavefunction values
        """
        N = self.n_points
        dx = self.dx
        dy = self.dy
        
        # Create 1D operators
        diagonals = [1, -2, 1]
        positions = [-1, 0, 1]
        D2 = diags(diagonals, positions, shape=(N, N)) * (-hbar**2 / (2 * m_e * dx**2))
        I = diags(np.ones(N), 0)
        
        # Create 2D Hamiltonian
        H = kron(I, D2) + kron(D2, I)
        
        # Add potential
        V_matrix = diags(self.V, 0)
        H = H + V_matrix
        
        # Solve eigenvalue problem
        energies, states = eigsh(H, k=n_eigenstates, which='SA')
        
        # Reshape wavefunctions to 2D
        wavefunctions = [state.reshape(N, N) for state in states.T]
        
        # Normalize wavefunctions
        for i in range(n_eigenstates):
            norm = simpson(y=simpson(y=np.abs(wavefunctions[i])**2, x=self.x, axis=1), x=self.y)
            wavefunctions[i] /= np.sqrt(norm)
        
        return energies, wavefunctions

def find_energy_levels(qdot: Union[QuantumDot1D, QuantumDot2D], 
                      E_min: float, E_max: float, 
                      n_levels: int = 5, tol: float = 1e-3) -> List[float]:
    """
    Find energy levels using bisection method.
    
    Parameters:
    -----------
    qdot : QuantumDot1D or QuantumDot2D
        Quantum dot instance
    E_min : float
        Minimum energy to search (eV)
    E_max : float
        Maximum energy to search (eV)
    n_levels : int
        Number of energy levels to find
    tol : float
        Tolerance for energy level convergence
    """
    def count_nodes(psi):
        """Count the number of nodes in a wavefunction"""
        return len(np.where(np.diff(np.sign(psi)))[0])
    
    energy_levels = []
    for n in range(n_levels):
        E_low = E_min
        E_high = E_max
        
        while E_high - E_low > tol:
            E_mid = (E_low + E_high) / 2
            _, psi = qdot.numerical_solution(E_mid)
            nodes = count_nodes(psi)
            
            if nodes < n:
                E_low = E_mid
            else:
                E_high = E_mid
        
        energy_levels.append((E_low + E_high) / 2)
    
    return energy_levels

def main():
    """Create and display interactive quantum dot visualization."""
    # Set up parameters
    width = 10e-9  # 10 nm
    depth = 0.3 * eV  # 0.3 eV
    n_points = 1000
    
    # Create quantum dots with different potentials
    quantum_dots = {
        'square': QuantumDot1D(width, depth, n_points, 'square'),
        'harmonic': QuantumDot1D(width, depth, n_points, 'harmonic'),
        'gaussian': QuantumDot1D(width, depth, n_points, 'gaussian')
    }
    
    # Create 2D quantum dot
    qdot2d = QuantumDot2D(width, depth, n_points=50)
    
    # Create interactive dashboard
    fig1, anim1 = create_interactive_dashboard(quantum_dots)
    
    # Create 2D animation in a separate window
    fig2, anim2 = plot_animated_2d_evolution(qdot2d)
    
    # Show both figures
    plt.show()

if __name__ == '__main__':
    main() 