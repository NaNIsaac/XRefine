import torch

from xrefine.data.generator import generate_synthetic_batch

def test_generator():
    print("Testing Data Generator...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 4
    q_grid = torch.linspace(0.01, 0.5, 100).to(device)
    
    config = {
        'n_layers': 2,
        'thickness_min': 50.0, 'thickness_max': 200.0,
        'roughness_min': 1.0, 'roughness_max': 5.0,
        'sld_min': 10.0, 'sld_max': 50.0,
        'intensity_counts': 1e7,
        'dq': 0.01,
        'sample_len': 10.0,
        'wavelength': 1.54,
        'beam_width': 0.2,
        'apply_footprint': True,
        'background_signal': 1e-6,
        'noise_scale': 1.0 # purely for documentation, generator logic uses poisson
    }
    
    inputs, (thick, rough, sld) = generate_synthetic_batch(batch_size, q_grid, config)
    
    print(f"Inputs shape: {inputs.shape}")
    print(f"Thickness shape: {thick.shape}")
    print(f"Roughness shape: {rough.shape}")
    print(f"SLD shape: {sld.shape}")
    
    assert inputs.shape == (batch_size, 100)
    assert thick.shape == (batch_size, 2)
    assert rough.shape == (batch_size, 3) # Amb/L1, L1/L2, L2/Sub
    assert sld.shape == (batch_size, 4)   # Amb, L1, L2, Sub
    
    print("Data ranges:")
    print(f"Reflectivity: {inputs.min().item():.3e} - {inputs.max().item():.3e}")
    
    # Check for NaN
    assert not torch.isnan(inputs).any()
    
    print("Generator verification passed!")

if __name__ == "__main__":
    test_generator()
