import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

class NCFODE_Enhancer:
    def __init__(self, alpha=0.85, lambda_reg=0.1, delta_s=0.1, epsilon=1e-8):
        """
        Initialize the NC-FODE parameters.
        alpha: Fractional order (Paper uses 0.85 as per Table 2)
        lambda_reg: Regularization parameter balancing data fidelity and smoothing
        delta_s: Step size for the iteration
        epsilon: Zero-point denominator to prevent division by zero errors
        """
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.delta_s = delta_s
        self.eps = epsilon

    def _fractional_derivative_2d(self, image, order):
        """
        Computes the fractional derivative of a 2D image using the Fourier transform.
        D^a f(x) = IFFT( (i*w)^a * FFT(f(x)) )
        """
        rows, cols = image.shape
        
        # Create frequency grid
        u = np.fft.fftfreq(rows)
        v = np.fft.fftfreq(cols)
        U_freq, V_freq = np.meshgrid(u, v, indexing='ij')
        
        # Frequency domain representation (omega)
        omega_x = 2 * np.pi * U_freq
        omega_y = 2 * np.pi * V_freq
        
        # Transform image to frequency domain
        img_fft = fft2(image)
        
        # Apply fractional derivative multiplier (i*omega)^alpha
        multiplier_x = (1j * omega_x) ** order
        multiplier_y = (1j * omega_y) ** order
        
        # Inverse transform back to spatial domain
        grad_x = np.real(ifft2(img_fft * multiplier_x))
        grad_y = np.real(ifft2(img_fft * multiplier_y))
        
        return grad_x, grad_y

    def _fractional_divergence(self, fx, fy, order):
        """
        Computes the fractional divergence of a 2D vector field (fx, fy).
        """
        div_x, _ = self._fractional_derivative_2d(fx, order)
        _, div_y = self._fractional_derivative_2d(fy, order)
        return div_x + div_y

    def enhance_image(self, original_image, iterations=15):
        """
        The main NC-FODE iterative loop for pixel strength computation.
        original_image: The degraded/noisy input image (F)
        iterations: Number of steps to evolve the differential equation
        """
        # Ensure image is float and normalized to [0, 1] to prevent overflow
        F = original_image.astype(np.float64)
        if np.max(F) > 1.0:
            F = F / 255.0
            
        # Initialize U (the evolving image) as the original image
        U = np.copy(F)
        
        # Constant multiplier from the equation: -(-1)^a
        # Since 'a' is fractional, this is complex. We take the real part for image processing.
        phase_factor = np.real(-(-1 + 0j)**self.alpha)
        
        for i in range(iterations):
            # 1. Compute fractional gradients: nabla^a U
            grad_x, grad_y = self._fractional_derivative_2d(U, self.alpha)
            
            # 2. Magnitude of the fractional gradient: |nabla^a U|
            # Added epsilon here to prevent the zero-division error mentioned in the paper
            grad_mag = np.sqrt(grad_x**2 + grad_y**2) + self.eps
            
            # 3. Normalize gradients: (nabla^a U) / |nabla^a U|
            norm_grad_x = grad_x / grad_mag
            norm_grad_y = grad_y / grad_mag
            
            # 4. Compute fractional divergence of the normalized gradients
            div_term = self._fractional_divergence(norm_grad_x, norm_grad_y, self.alpha)
            
            # 5. Data fidelity term: lambda * (1 - F/U)
            # Add epsilon to U to prevent division by zero in dark regions
            fidelity_term = self.lambda_reg * (1.0 - (F / (U + self.eps)))
            
            # 6. Full Update Equation (Equation 13)
            # U^(b+1) = U^b + delta_s * [ phase_factor * (1/U^b) * div_term - fidelity_term ]
            update = self.delta_s * (
                phase_factor * (1.0 / (U + self.eps)) * div_term - fidelity_term
            )
            
            U = U + update
            
            # Clip values to maintain valid image range
            U = np.clip(U, 0.0, 1.0)
            
        return U