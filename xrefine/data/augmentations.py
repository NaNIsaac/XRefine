import numpy as np
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d

class XRRAugmentations:
    """
    Physics-based data augmentation for XRR.
    
    Features:
    - Geometric Footprint correction
    - Instrument Resolution smearing (Gaussian convolution)
    - Background & Poison Noise
    - Q-axis misalignment (Shift)
    """
    def __init__(
        self,
        intensity_noise_scale: float = 0.2,
        bg_range: tuple[float, float] = (-8, -5),
        res_sigma_range: tuple[float, float] = (0.0001, 0.005),
        q_shift_sigma: float = 0.002,
        delta_q: float = 0.0005,
        prob: float = 0.95,
        beam_width_range: tuple = (0.05, 0.2), # Adjusted to realistic mm range (similar to config 0.1)
        sample_len_range: tuple = (5, 30.0),
        do_footprint: bool = True,
        beam_profile: str = 'gaussian' # 'gaussian' or 'square'
    ):
        self.intensity_noise_scale = intensity_noise_scale
        self.bg_range = bg_range
        self.res_sigma_range = res_sigma_range
        self.q_shift_sigma = q_shift_sigma
        self.delta_q = delta_q
        self.prob = prob

        # Footprint settings
        self.beam_width_range = beam_width_range
        self.sample_len_range = sample_len_range
        self.do_footprint = do_footprint
        self.beam_profile = beam_profile

    def _gauss_footprint(self, theta_deg: np.ndarray, beam_w: float, sample_len: float) -> np.ndarray:
        """
        GenX's GaussIntensity logic implemented for NumPy.
        beam_w is assumed to be FWHM.
        """
        rad = np.pi / 180.0
        sqrt2 = np.sqrt(2.0)
        
        # Convert FWHM to Sigma
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma
        sigma_x = beam_w / (2 * np.sqrt(2 * np.log(2)))

        # 빔과 시료가 이루는 투영 각도 계산
        sinalpha = np.sin(theta_deg * rad)

        # 0으로 나누기 방지 (theta=0일 때)
        sinalpha = np.maximum(sinalpha, 1e-9)

        s_half = sample_len / 2.0

        # Gaussian Beam Spill-over Calculation
        common = sinalpha / (sqrt2 * sigma_x)

        # erf is odd function: erf(s * common) + erf(s * common) = 2 * erf(...)
        # result = (2 * erf(...)) / 2.0 = erf(...)
        correction = erf(s_half * common)

        return correction

    def _square_footprint(self, theta_deg: np.ndarray, beam_w: float, sample_len: float) -> np.ndarray:
        """
        GenX's SquareIntensity logic.
        """
        rad = np.pi / 180.0
        sinalpha = np.sin(theta_deg * rad)
        
        # F = slen / beamwidth * sin(alpha)
        # GenX uses 'beamwidth' directly for Square profile
        F = (sample_len / beam_w) * sinalpha
        
        return np.where(F <= 1.0, F, np.ones_like(F))

    def apply_footprint(self, q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Randomly apply beam width and sample length for low-angle intensity attenuation simulation."""
        beam_w = np.random.uniform(*self.beam_width_range)
        sample_len = np.random.uniform(*self.sample_len_range)

        wavelength = 1.54
        sin_theta = q * wavelength / (4 * np.pi)
        sin_theta = np.clip(sin_theta, 0, 1)
        theta_deg = np.degrees(np.arcsin(sin_theta))

        if self.beam_profile == 'square':
            foocor = self._square_footprint(theta_deg, beam_w, sample_len)
        else:
            foocor = self._gauss_footprint(theta_deg, beam_w, sample_len)

        return R * foocor

    def __call__(self, q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Probability check
        if np.random.rand() > self.prob:
            return q, R

        R_aug = R.copy()

        # [Step 1] Geometric Footprint
        if self.do_footprint:
            R_aug = self.apply_footprint(q, R_aug)

        # [Step 2] Instrument Resolution Smearing
        # Ensure delta_q is valid; if 0 (e.g. single point), skip smearing
        if self.delta_q > 0:
            res_sigma_q = np.random.uniform(*self.res_sigma_range)
            sigma_px = res_sigma_q / self.delta_q

            if sigma_px > 0:
                R_aug = gaussian_filter1d(R_aug, sigma=sigma_px, mode="nearest")

        # [Step 3] Detectors & Electronics (Noise & Scale)
        
        # Global Scaling start
        scale = np.random.uniform(1 - self.intensity_noise_scale, 1 + self.intensity_noise_scale)
        R_aug *= scale

        # Background Noise
        bg_level = np.random.uniform(*self.bg_range)
        # Generate background
        b = pow(10, np.random.uniform(bg_level, bg_level + 0.5)) 
        bg = np.random.normal(b, 0.1 * b, len(R_aug))
        
        R_aug = R_aug + np.abs(bg)

        # Poisson Noise (Shot noise)
        s_rand = pow(10, np.random.uniform(5, 8))
        # Poisson noise implementation
        expected_counts = s_rand * R_aug
        # Fix: ensure non-negative
        expected_counts = np.maximum(expected_counts, 0)
        noisy_counts = np.random.poisson(expected_counts)
        R_aug = noisy_counts / s_rand

        # [Step 4] Q-axis Misalignment
        if self.q_shift_sigma > 0:
            q_shift = np.random.normal(0, self.q_shift_sigma)
            R_aug = np.interp(q, q + q_shift, R_aug, left=R_aug[0], right=R_aug[-1])

        return q, R_aug
