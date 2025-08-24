import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.animation as animation
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - using NumPy")

## Constants and Parameters 
c = 299792458.0  # Exact speed of light (m/s)
h = 6.62607015e-34  # Planck constant (J/Hz)
hbar = h / (2 * np.pi)
ε0 = 8.8541878128e-12  # Vacuum permittivity

# Simulation parameters
wavelength = 632.8e-9  # HeNe laser wavelength (m)
k = 2 * np.pi / wavelength  # Wavenumber
z0 = 0.15  # Propagation distance (m)
L = 0.03  # Side length of hologram plane (m)
N = 256 if GPU_AVAILABLE else 128  # Reduced for animation speed here
dt = 1e-15  # Time step for animation (s)

# Create coordinate system
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
fx = fftshift(np.fft.fftfreq(N, d=L/N))
fy = fftshift(np.fft.fftfreq(N, d=L/N))
FX, FY = np.meshgrid(fx, fy)

## RelativisticObject class (unchanged except slight N reduction)
class RelativisticObject:
    def __init__(self, shape, position, size, velocity, material=None):
        self.shape = shape.lower()
        self.position = np.array(position, dtype=np.float64)
        self.size = size
        self.velocity = np.array(velocity, dtype=np.float64)
        self.material = material or {'n': 1.0, 'μ': 1.0}  # Refractive index, permeability
        self.quantum_state = None
        self.points = self._generate_points()
        self._setup_quantum_state()
        
    def _generate_points(self):
        np.random.seed(42)
        if self.shape == 'hypercube':
            points = np.random.uniform(-self.size/2, self.size/2, (4, 1000))
            w = points[3]
            points = points[:3] / (1 + w[:, None]/self.size)
            points = points.T
        elif self.shape == 'spacetime-torus':
            theta = np.linspace(0, 2*np.pi, 50)
            phi = np.linspace(0, 2*np.pi, 25)
            theta, phi = np.meshgrid(theta, phi)
            R = self.size
            r = R/3
            x = (R + r*np.cos(theta)) * np.cos(phi)
            y = (R + r*np.cos(theta)) * np.sin(phi)
            z = r * np.sin(theta)
            points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        else:
            num_points = 1500
            indices = np.arange(0, num_points, dtype=float) + 0.5
            phi = np.arccos(1 - 2*indices/num_points)
            theta = np.pi * (1 + 5**0.5) * indices
            x = np.cos(theta) * np.sin(phi) * self.size
            y = np.sin(theta) * np.sin(phi) * self.size
            z = np.cos(phi) * self.size
            points = np.vstack([x, y, z]).T
        return (points + self.position).T
    
    def _setup_quantum_state(self):
        num_points = self.points.shape[1]
        self.quantum_state = {
            'phase': np.random.uniform(0, 2*np.pi, num_points),
            'entanglement': np.ones((num_points, num_points)),
            'polarization': np.random.uniform(0, np.pi/2, num_points)
        }
    
    def lorentz_transform(self, beta):
        gamma = 1/np.sqrt(1 - beta**2)
        v = beta * c
        x, y, z = self.points
        t = np.zeros_like(x)
        x_prime = gamma*(x - v*t)
        y_prime = y
        z_prime = z
        # Length contraction approx:
        x_prime /= gamma
        # Doppler shift quantum phases
        if self.quantum_state:
            doppler_factor = np.sqrt((1 + beta)/(1 - beta))
            self.quantum_state['phase'] *= doppler_factor
        return x_prime, y_prime, z_prime
    
    def calculate_optical_response(self, wavelength):
        n = self.material['n']
        k_material = 2 * np.pi * n / wavelength
        r = (1 - n)/(1 + n)
        t = 2/(1 + n)
        return {
            'reflectivity': np.abs(r)**2,
            'transmissivity': np.abs(t)**2,
            'phase_shift': k_material * self.size
        }

## HolographySimulator class (simplified & adapted)
class HolographySimulator:
    def __init__(self, objects, wavelength, N=128):
        self.objects = objects
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.N = N
        self.x = np.linspace(-L/2, L/2, N)
        self.y = np.linspace(-L/2, L/2, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.beta = 0.0
        if GPU_AVAILABLE:
            self._X = cp.asarray(self.X)
            self._Y = cp.asarray(self.Y)
            self._hologram = cp.zeros((self.N, self.N), dtype=cp.complex128)
        else:
            self._hologram = np.zeros((self.N, self.N), dtype=np.complex128)
    
    def generate_hologram(self, beta):
        self.beta = beta
        reference_wave = np.exp(1j * self.k * self.X)
        if GPU_AVAILABLE:
            self._hologram.fill(0)
            _reference = cp.asarray(reference_wave)
        else:
            self._hologram.fill(0)
        
        for obj in self.objects:
            xp, yp, zp = obj.lorentz_transform(beta)
            mat_response = obj.calculate_optical_response(self.wavelength)
            for i in range(xp.shape[0]):
                r = np.sqrt((self.X - xp[i])**2 + (self.Y - yp[i])**2 + zp[i]**2)
                phase = obj.quantum_state['phase'][i] if obj.quantum_state else 0
                pol_angle = obj.quantum_state['polarization'][i] if obj.quantum_state else 0
                pol_factor = np.cos(pol_angle)**2
                material_phase = mat_response['phase_shift']
                wave = (pol_factor * mat_response['transmissivity'] * 
                        np.exp(1j*(self.k*(self.X - xp[i]) + self.k*(self.Y - yp[i]) + phase + material_phase))) / (r + 1e-9)
                if GPU_AVAILABLE:
                    wave_gpu = cp.asarray(wave)
                    self._hologram += wave_gpu
                else:
                    self._hologram += wave
        
        if GPU_AVAILABLE:
            hologram = cp.asnumpy(cp.abs(_reference + self._hologram)**2)
        else:
            hologram = np.abs(reference_wave + self._hologram)**2
            
        return hologram / np.max(hologram)
    
    def reconstruct(self, hologram, z):
        k = 2 * np.pi / self.wavelength
        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY
        kz = np.sqrt(k**2 - kx**2 - ky**2 + 0j)
        H = np.exp(1j * kz * z)
        H[np.sqrt(kx**2 + ky**2) > k] = 0
        if GPU_AVAILABLE:
            _hologram = cp.asarray(hologram * np.exp(1j * self.k * self.X))
            _H = cp.asarray(H)
            recon = cp.asnumpy(ifft2(fft2(_hologram) * _H))
        else:
            recon = np.abs(ifft2(fft2(hologram * np.exp(1j * self.k * self.X)) * H))
        return recon / np.max(recon)

## Quantum Information Metrics
def calculate_entanglement_entropy(objects):
    entropy = 0
    for obj in objects:
        if obj.quantum_state:
            rho = obj.quantum_state['entanglement']
            eigenvalues = np.linalg.eigvalsh(rho)
            entropy += -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))
    return entropy

## InteractiveVisualizer class with animation
class InteractiveVisualizer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.fig = plt.figure(figsize=(18, 12), facecolor='black')
        self._setup_ui()
        self._create_plots()
        self.update()
        
    def _setup_ui(self):
        self.beta_slider = Slider(plt.axes([0.15, 0.02, 0.3, 0.03], facecolor='lightgray'),
                                  'β (v/c)', 0.0, 0.99, valinit=0.0, valstep=0.01)
        self.z_slider = Slider(plt.axes([0.55, 0.02, 0.3, 0.03], facecolor='lightgray'),
                               'Reconstruction Distance (m)', 0.05, 0.3, valinit=z0, valstep=0.01)
        self.reset_btn = Button(plt.axes([0.05, 0.02, 0.08, 0.03]), 'Reset', color='lightgray')
        self.animate_btn = Button(plt.axes([0.05, 0.06, 0.08, 0.03]), 'Animate', color='lightgray')
        self.object_radio = RadioButtons(plt.axes([0.88, 0.7, 0.1, 0.2]), ['Hypercube', 'Torus', 'Sphere'], active=2)
        self.beta_slider.on_changed(self.update)
        self.z_slider.on_changed(self.update)
        self.reset_btn.on_clicked(self.reset)
        self.animate_btn.on_clicked(self.animate)
        self.object_radio.on_clicked(self.change_object)

    def _create_plots(self):
        self.ax1 = self.fig.add_subplot(241, projection='3d')
        self.ax1.set_title("Rest Frame Objects", color='white')
        self.ax2 = self.fig.add_subplot(242, projection='3d')
        self.ax2.set_title("Moving Frame (β=0.0)", color='white')
        self.ax3 = self.fig.add_subplot(243)
        self.holo_img = self.ax3.imshow(np.zeros((N, N)), cmap='viridis')
        self.ax3.set_title("Hologram", color='white')
        self.ax4 = self.fig.add_subplot(244)
        self.recon_img = self.ax4.imshow(np.zeros((N, N)), cmap='inferno')
        self.ax4.set_title("Reconstruction", color='white')
        self.ax5 = self.fig.add_subplot(245)
        self.ax5.axis('off')
        self.metrics_text = self.ax5.text(0.5, 0.5, "", ha='center', va='center', color='white', fontsize=10)
        self.ax6 = self.fig.add_subplot(246)
        self.fourier_img = self.ax6.imshow(np.zeros((N, N)), cmap='plasma')
        self.ax6.set_title("Fourier Spectrum", color='white')
        self.ax7 = self.fig.add_subplot(247)
        self.corr_img = self.ax7.imshow(np.zeros((N, N)), cmap='coolwarm')
        self.ax7.set_title("Spacetime Correlation", color='white')
        self.ax8 = self.fig.add_subplot(248)
        self.phase_img = self.ax8.imshow(np.zeros((N, N)), cmap='hsv')
        self.ax8.set_title("Quantum Phase", color='white')

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6, self.ax7, self.ax8]:
            ax.set_facecolor('black')
            ax.xaxis.set_tick_params(color='white', labelcolor='white')
            ax.yaxis.set_tick_params(color='white', labelcolor='white')
        plt.colorbar(self.holo_img, ax=self.ax3, label='Intensity')
        plt.colorbar(self.recon_img, ax=self.ax4, label='Intensity')
        plt.colorbar(self.fourier_img, ax=self.ax6, label='Magnitude')
        plt.colorbar(self.corr_img, ax=self.ax7, label='Correlation')
        plt.colorbar(self.phase_img, ax=self.ax8, label='Phase (rad)')

    def update(self, val=None):
        beta = self.beta_slider.val
        z = self.z_slider.val
        start_time = time.time()
        hologram = self.simulator.generate_hologram(beta)
        reconstruction = self.simulator.reconstruct(hologram, z)
        self._update_3d_views(beta)
        self.holo_img.set_data(hologram)
        self.holo_img.set_clim(0, hologram.max())
        self.recon_img.set_data(reconstruction)
        self.recon_img.set_clim(0, reconstruction.max())
        spectrum = np.log(1 + np.abs(fftshift(fft2(hologram))))
        self.fourier_img.set_data(spectrum)
        self.fourier_img.set_clim(spectrum.min(), spectrum.max())
        entropy = calculate_entanglement_entropy(self.simulator.objects)
        self.metrics_text.set_text(
            f"β = {beta:.3f}\n"
            f"γ = {1/np.sqrt(1 - beta**2):.3f}\n"
            f"Entanglement Entropy = {entropy:.3f}\n"
            f"Compute Time = {time.time()-start_time:.2f}s\n"
            f"GPU Acceleration = {GPU_AVAILABLE}"
        )
        phase = np.angle(fft2(hologram))
        self.phase_img.set_data(phase)
        self.fig.canvas.draw_idle()

    def _update_3d_views(self, beta):
        self.ax1.clear()
        self.ax2.clear()
        for obj in self.simulator.objects:
            x, y, z = obj.points
            self.ax1.scatter(x, y, z, s=1, alpha=0.6, label=obj.shape)
            xp, yp, zp = obj.lorentz_transform(beta)
            self.ax2.scatter(xp, yp, zp, s=1, alpha=0.6)
        self.ax1.set_title("Rest Frame Objects", color='white')
        self.ax2.set_title(f"Moving Frame (β={beta:.2f})", color='white')
        self.ax1.legend()
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('black')
            ax.xaxis.set_tick_params(color='white', labelcolor='white')
            ax.yaxis.set_tick_params(color='white', labelcolor='white')
            ax.zaxis.set_tick_params(color='white', labelcolor='white')

    def reset(self, event):
        self.beta_slider.set_val(0.0)
        self.z_slider.set_val(z0)
        self.update()

    def animate(self, event=None):
        # Use FuncAnimation for smooth animation
        self.ani = animation.FuncAnimation(
            self.fig, self._animate_func, frames=100, interval=100, blit=False, repeat=True)
        plt.show()

    def _animate_func(self, frame):
        beta = 0.95 * (frame / 99)
        self.beta_slider.set_val(beta)
        self.update()
        return []

    def change_object(self, label):
        new_shape = label.lower()
        for obj in self.simulator.objects:
            obj.shape = new_shape
            obj.points = obj._generate_points()
        self.update()

## Main Execution 
if __name__ == "__main__":
    # Create objects
    objects = [
        RelativisticObject('sphere', [-0.005, 0.005, 0.01], 0.002, [0.5*c, 0, 0]),
        RelativisticObject('sphere', [0.005, 0.005, 0.01], 0.002, [-0.7*c, 0, 0]),
    ]
    simulator = HolographySimulator(objects, wavelength, N)
    visualizer = InteractiveVisualizer(simulator)
    
    # Start animation immediately
    visualizer.animate()
    
ani = FuncAnimation(fig, update, frames=len(beta_values), interval=200, blit=True)
plt.close()