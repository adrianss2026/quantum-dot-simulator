import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec

# Physical constants
hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
m_e = 9.10938356e-31  # Electron mass (kg)
eV = 1.602176634e-19  # Electron volt (J)
e = 1.602176634e-19   # Elementary charge (C)
epsilon_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)

# Copyright (c) 2024 Adrian S
# This file is part of the Quantum Dot Modeling Project and is licensed under the MIT License.
# See LICENSE file for details.

def create_custom_colormap():
    """Create a custom colormap for quantum mechanical visualizations"""
    colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    return LinearSegmentedColormap.from_list('quantum', colors)

def plot_wavefunction(x: np.ndarray, psi: np.ndarray, title: str,
                     potential: Optional[np.ndarray] = None,
                     energy: Optional[float] = None):
    """
    Plot the wavefunction with enhanced visualization.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Position array
    psi : numpy.ndarray
        Wavefunction values
    title : str
        Plot title
    potential : numpy.ndarray, optional
        Potential energy values
    energy : float, optional
        Energy level
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scale the potential for better visualization
    V_scaled = potential / np.max(np.abs(potential)) * np.max(np.abs(psi))
    
    # Plot potential
    ax.plot(x, V_scaled, 'k--', label='Potential (scaled)')
    
    # Plot real and imaginary parts of wavefunction
    ax.plot(x, np.real(psi), 'b-', label='Re(ψ)')
    ax.plot(x, np.imag(psi), 'r-', label='Im(ψ)')
    
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig, ax

def plot_probability_density(x: np.ndarray, psi: np.ndarray, title: str,
                           potential: Optional[np.ndarray] = None,
                           energy: Optional[float] = None):
    """
    Plot the probability density with enhanced visualization.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Position array
    psi : numpy.ndarray
        Wavefunction values
    title : str
        Plot title
    potential : numpy.ndarray, optional
        Potential energy values
    energy : float, optional
        Energy level
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scale the potential for better visualization
    V_scaled = potential / np.max(np.abs(potential)) * np.max(np.abs(psi)**2)
    
    # Plot potential
    ax.plot(x, V_scaled, 'k--', label='Potential (scaled)')
    
    # Plot probability density
    ax.plot(x, np.abs(psi)**2, 'g-', label='|ψ|²')
    
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig, ax

def plot_potential(x, V, title="Potential Energy"):
    """
    Plot the potential energy profile.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Position array
    V : numpy.ndarray
        Potential energy values
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x * 1e9, V, 'g-', linewidth=2)  # Convert x to nm
    plt.xlabel('Position (nm)')
    plt.ylabel('Potential Energy (eV)')
    plt.title(title)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_2d_wavefunction(X: np.ndarray, Y: np.ndarray, psi: np.ndarray,
                        title: str, potential: Optional[np.ndarray] = None,
                        energy: Optional[float] = None):
    """
    Plot 2D wavefunction using surface and contour plots with enhanced visualization.
    
    Parameters:
    -----------
    X : numpy.ndarray
        X-coordinate meshgrid
    Y : numpy.ndarray
        Y-coordinate meshgrid
    psi : numpy.ndarray
        2D wavefunction values
    title : str
        Plot title
    potential : numpy.ndarray, optional
        Potential energy values
    energy : float, optional
        Energy level
    """
    cmap = create_custom_colormap()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 8))
    
    # Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X * 1e9, Y * 1e9, psi, cmap=cmap,
                          linewidth=0, antialiased=True)
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.set_zlabel('Wavefunction')
    ax1.set_title(f'{title} - Surface Plot')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X * 1e9, Y * 1e9, psi, levels=20, cmap=cmap)
    ax2.set_xlabel('X (nm)')
    ax2.set_ylabel('Y (nm)')
    ax2.set_title(f'{title} - Contour Plot')
    fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
    
    if energy is not None:
        fig.suptitle(f'{title} (Energy: {energy:.3f} eV)')
    
    plt.tight_layout()

def plot_2d_probability_density(X: np.ndarray, Y: np.ndarray, psi: np.ndarray,
                              title: str, potential: Optional[np.ndarray] = None,
                              energy: Optional[float] = None):
    """
    Plot 2D probability density with enhanced visualization.
    
    Parameters:
    -----------
    X : numpy.ndarray
        X-coordinate meshgrid
    Y : numpy.ndarray
        Y-coordinate meshgrid
    psi : numpy.ndarray
        2D wavefunction values
    title : str
        Plot title
    potential : numpy.ndarray, optional
        Potential energy values
    energy : float, optional
        Energy level
    """
    probability = np.abs(psi)**2
    cmap = create_custom_colormap()
    
    plt.figure(figsize=(12, 10))
    plt.contourf(X * 1e9, Y * 1e9, probability, levels=20, cmap=cmap)
    plt.colorbar(label='Probability Density')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    
    if energy is not None:
        plt.title(f'{title} (Energy: {energy:.3f} eV)')
    else:
        plt.title(title)
    
    plt.tight_layout()

def plot_energy_levels(energy_levels: List[float], title: str = "Energy Levels",
                      potential_type: Optional[str] = None):
    """
    Plot energy levels with enhanced visualization.
    
    Parameters:
    -----------
    energy_levels : list
        List of energy levels in eV
    title : str
        Plot title
    potential_type : str, optional
        Type of potential profile
    """
    plt.figure(figsize=(12, 8))
    
    # Plot energy levels
    for i, E in enumerate(energy_levels):
        plt.axhline(y=E, color='b', linestyle='-', alpha=0.5)
        plt.text(0.1, E, f'E{i+1} = {E:.3f} eV',
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.xlabel('Quantum State')
    plt.ylabel('Energy (eV)')
    
    if potential_type:
        plt.title(f'{title} - {potential_type.capitalize()} Potential')
    else:
        plt.title(title)
    
    plt.grid(True)
    plt.tight_layout()

def plot_convergence(n_points: np.ndarray, energies: np.ndarray,
                    title: str = "Convergence Analysis"):
    """
    Plot convergence analysis results.
    
    Parameters:
    -----------
    n_points : numpy.ndarray
        Number of grid points
    energies : numpy.ndarray
        Corresponding energy values
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    plt.loglog(n_points, np.abs(energies - energies[-1]), 'b-o',
              linewidth=2, markersize=8)
    plt.xlabel('Number of Grid Points')
    plt.ylabel('Energy Error (eV)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

def plot_potential_comparison(x: np.ndarray, potentials: List[Tuple[np.ndarray, str]],
                            title: str = "Potential Comparison"):
    """
    Plot comparison of different potential profiles.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Position array
    potentials : list of tuples
        List of (potential_values, label) pairs
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for V, label in potentials:
        plt.plot(x * 1e9, V, label=label, linewidth=2)
    
    plt.xlabel('Position (nm)')
    plt.ylabel('Potential Energy (eV)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def animate_wavefunction(x, psi_t, potential, dt, interval=50):
    """
    Animate the time evolution of a wavefunction.
    
    Parameters:
    -----------
    x : array_like
        Spatial coordinates
    psi_t : function
        Function that returns the wavefunction at time t
    potential : array_like
        Potential energy
    dt : float
        Time step
    interval : int
        Animation interval in milliseconds
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scale the potential for better visualization
    V_scaled = potential / np.max(np.abs(potential)) * np.max(np.abs(psi_t(0)))
    
    # Plot potential
    ax.plot(x, V_scaled, 'k--', label='Potential (scaled)')
    
    # Initialize the lines
    line_real, = ax.plot([], [], 'b-', label='Re(ψ)')
    line_imag, = ax.plot([], [], 'r-', label='Im(ψ)')
    line_prob, = ax.plot([], [], 'g-', alpha=0.3, label='|ψ|²')
    
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-1.5, 1.5)
    ax.legend()
    ax.grid(True)
    
    def init():
        line_real.set_data([], [])
        line_imag.set_data([], [])
        line_prob.set_data([], [])
        return line_real, line_imag, line_prob
    
    def animate(frame):
        t = frame * dt
        psi = psi_t(t)
        
        line_real.set_data(x, np.real(psi))
        line_imag.set_data(x, np.imag(psi))
        line_prob.set_data(x, np.abs(psi)**2)
        
        ax.set_title(f'Time: {t:.2e} s')
        return line_real, line_imag, line_prob
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=200,
                        interval=interval, blit=True)
    
    return anim

def plot_energy_levels(energies, potentials, labels=None):
    """Plot energy levels for different potentials."""
    if labels is None:
        labels = [f'Potential {i+1}' for i in range(len(potentials))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (E, V, label) in enumerate(zip(energies, potentials, labels)):
        # Plot potential
        ax.plot(V, 'k--', alpha=0.3)
        
        # Plot energy levels
        for j, energy in enumerate(E):
            ax.axhline(y=energy, color=f'C{i}', linestyle='-',
                      label=f'{label} n={j}' if j == 0 else None)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Energy Levels')
    ax.legend()
    ax.grid(True)
    
    return fig, ax

def create_interactive_dashboard(quantum_dots, fig_size=(15, 10)):
    """
    Create an interactive dashboard for quantum dot visualization.
    
    Parameters:
    -----------
    quantum_dots : dict
        Dictionary containing quantum dot objects for different potentials
    fig_size : tuple
        Figure size (width, height)
    """
    # Use Qt5Agg backend for better animation support
    import matplotlib
    matplotlib.use('Qt5Agg')
    
    # Create figure and grid layout
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(3, 3)
    
    # Create subplots
    ax_wave = fig.add_subplot(gs[0, :2])  # Wavefunction plot
    ax_prob = fig.add_subplot(gs[1, :2])  # Probability density plot
    ax_energy = fig.add_subplot(gs[2, :2]) # Energy levels
    ax_potential = fig.add_subplot(gs[:, 2]) # Potential plot
    
    # Get initial quantum dot
    qdot = quantum_dots['square']
    x = qdot.x
    
    # Initialize plots with actual data
    E, psi = qdot.numerical_solution(1)
    
    lines_wave = {}
    lines_prob = {}
    for pot_type in quantum_dots:
        lines_wave[pot_type], = ax_wave.plot(x, np.zeros_like(x), 
                                           label=f'{pot_type.capitalize()} Re(ψ)')
        lines_prob[pot_type], = ax_prob.plot(x, np.zeros_like(x), 
                                           label=f'{pot_type.capitalize()} |ψ|²')
    
    # Set up axes
    ax_wave.set_xlim(x[0], x[-1])
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_xlabel('Position (m)')
    ax_wave.set_ylabel('Wavefunction')
    ax_wave.legend()
    ax_wave.grid(True)
    
    ax_prob.set_xlim(x[0], x[-1])
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel('Position (m)')
    ax_prob.set_ylabel('Probability Density')
    ax_prob.legend()
    ax_prob.grid(True)
    
    # Energy levels plot
    ax_energy.set_xlabel('State')
    ax_energy.set_ylabel('Energy (eV)')
    ax_energy.grid(True)
    
    # Potential plot
    ax_potential.plot(x, qdot.V, 'k--', label='Potential')
    ax_potential.set_xlabel('Position (m)')
    ax_potential.set_ylabel('Potential (J)')
    ax_potential.grid(True)
    ax_potential.legend()
    
    # Add sliders
    slider_ax_energy = plt.axes([0.15, 0.02, 0.3, 0.02])
    slider_ax_width = plt.axes([0.15, 0.06, 0.3, 0.02])
    
    slider_energy = Slider(slider_ax_energy, 'Energy (eV)', 0.01, 1.0, valinit=0.3)
    slider_width = Slider(slider_ax_width, 'Width (nm)', 1, 20, valinit=10)
    
    # Add radio buttons for potential selection
    radio_ax = plt.axes([0.85, 0.05, 0.1, 0.15])
    radio = RadioButtons(radio_ax, ('Square', 'Harmonic', 'Gaussian'))
    
    def update_plots(val=None):
        """Update all plots when parameters change"""
        potential_type = radio.value_selected.lower()
        qdot = quantum_dots[potential_type]
        
        # Update parameters
        qdot.depth = slider_energy.val * eV
        qdot.width = slider_width.val * 1e-9
        
        # Recalculate potential and wavefunctions
        qdot.V = qdot._create_potential()
        qdot.H = qdot._create_hamiltonian()
        
        # Calculate new states
        E, psi = qdot.numerical_solution(1)
        
        # Update wavefunction plot
        lines_wave[potential_type].set_data(qdot.x, np.real(psi[:, 0]))
        
        # Update probability plot
        lines_prob[potential_type].set_data(qdot.x, np.abs(psi[:, 0])**2)
        
        # Update potential plot
        ax_potential.clear()
        ax_potential.plot(qdot.x, qdot.V, 'k--', label='Potential')
        ax_potential.set_xlabel('Position (m)')
        ax_potential.set_ylabel('Potential (J)')
        ax_potential.grid(True)
        ax_potential.legend()
        
        fig.canvas.draw_idle()
    
    # Connect callbacks
    slider_energy.on_changed(update_plots)
    slider_width.on_changed(update_plots)
    radio.on_clicked(update_plots)
    
    def animate(frame):
        """Animation update function"""
        t = frame * 1e-15  # Time in femtoseconds
        potential_type = radio.value_selected.lower()
        qdot = quantum_dots[potential_type]
        
        # Get ground and excited states
        E0, psi0 = qdot.numerical_solution(1)  # Ground state
        E1, psi1 = qdot.numerical_solution(2)  # First excited state
        
        # Create superposition
        psi = (psi0[:, 0] * np.exp(-1j * E0[0] * t / hbar) +
               psi1[:, 0] * np.exp(-1j * E1[0] * t / hbar)) / np.sqrt(2)
        
        # Update plots
        lines_wave[potential_type].set_data(qdot.x, np.real(psi))
        lines_prob[potential_type].set_data(qdot.x, np.abs(psi)**2)
        
        return list(lines_wave.values()) + list(lines_prob.values())
    
    # Create animation with a shorter interval and fewer frames for better performance
    anim = FuncAnimation(fig, animate, frames=100, interval=100, blit=True)
    
    # Add play/pause button
    button_ax = plt.axes([0.5, 0.02, 0.1, 0.04])
    button = Button(button_ax, 'Play/Pause')
    
    def play_pause(event):
        if hasattr(anim, 'event_source') and anim.event_source:
            if anim.event_source.is_running():
                anim.event_source.stop()
            else:
                anim.event_source.start()
    
    button.on_clicked(play_pause)
    
    # Initial plot update
    update_plots()
    
    # Style and layout
    fig.suptitle('Interactive Quantum Dot Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    return fig, anim

def plot_animated_2d_evolution(quantum_dot_2d, n_frames=100, interval=100):
    """
    Create an animated plot of 2D quantum dot evolution.
    
    Parameters:
    -----------
    quantum_dot_2d : QuantumDot2D
        2D quantum dot object
    n_frames : int
        Number of animation frames
    interval : int
        Animation interval in milliseconds
    """
    # Calculate initial state
    E, psi = quantum_dot_2d.solve_2d(n_eigenstates=2)
    X, Y = quantum_dot_2d.X, quantum_dot_2d.Y
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize surface plot
    probability = np.abs(psi[0])**2
    surf = ax.plot_surface(X, Y, probability, cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('|ψ|²')
    ax.set_title('Quantum Dot Time Evolution')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, label='Probability Density')
    
    # Store the probability range for consistent scaling
    z_min = np.min(probability)
    z_max = np.max(probability)
    ax.set_zlim(z_min, z_max * 1.1)
    
    def update(frame):
        ax.clear()
        t = frame * 1e-15  # Time in femtoseconds
        
        # Create superposition state
        psi_t = (psi[0] * np.exp(-1j * E[0] * t / hbar) +
                 psi[1] * np.exp(-1j * E[1] * t / hbar)) / np.sqrt(2)
        
        # Calculate probability density
        probability = np.abs(psi_t)**2
        
        # Create new surface plot
        surf = ax.plot_surface(X, Y, probability, cmap='viridis')
        
        # Set labels and limits
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('|ψ|²')
        ax.set_title(f'Time: {t*1e15:.1f} fs')
        ax.set_zlim(z_min, z_max * 1.1)
        
        return [surf]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    
    # Add play/pause button
    button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(button_ax, 'Play/Pause')
    
    def play_pause(event):
        if hasattr(anim, 'event_source') and anim.event_source:
            if anim.event_source.is_running():
                anim.event_source.stop()
            else:
                anim.event_source.start()
    
    button.on_clicked(play_pause)
    
    plt.tight_layout()
    
    return fig, anim 