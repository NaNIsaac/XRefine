import torch
from xrefine.core.structures import Instrument, Sample, Layer
from xrefine.core.interface import Simulator
from xrefine.data.generator import generate_synthetic_batch

def test_structures():
    print("Testing Structures...")
    instr = Instrument(wavelength=1.54, I0=1e8)
    assert instr.wavelength == 1.54
    
    l1 = Layer(thickness=100.0, roughness=3.0, sld=complex(20.0, 0.0))
    s = Sample(layers=[l1])
    assert len(s.layers) == 1
    
    # Test to_tensor
    device = torch.device('cpu')
    t, r, sld = s.to_tensor(device)
    print(f"to_tensor shape: T={t.shape}, R={r.shape}, SLD={sld.shape}")
    
    assert t.shape == (1, 1)
    assert r.shape == (1, 2) # Amb/L1, L1/Sub
    assert sld.shape == (1, 3) # Amb, L1, Sub
    
    # Test from_tensor
    s_new = Sample.from_tensor(t[0], r[0], sld[0])
    assert len(s_new.layers) == 1
    assert abs(s_new.layers[0].thickness - 100.0) < 1e-5
    assert abs(s_new.layers[0].roughness - 3.0) < 1e-5
    assert abs(s_new.Substrate.roughness - 2.0) < 1e-5 # Default Si roughness
    
    print("Structures OK")

def test_simulator_with_sample():
    print("Testing Simulator with Sample object...")
    sim = Simulator(device='cpu')
    
    l1 = Layer(thickness=100.0, roughness=3.0, sld=complex(20.0, 0.0))
    # Ambient default AIR (rho=0), Substrate default Si (rho~20)
    s = Sample(layers=[l1])
    
    q = torch.linspace(0.01, 0.5, 50).unsqueeze(0)
    
    # We need to type hint Sample for internal method if we strictly check, 
    # but python is dynamic.
    r = sim.simulate_sample(q, s)
    
    assert r.shape == (1, 50)
    print("simulate_sample OK")

def test_generator_with_instrument():
    print("Testing generator with Instrument config...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_grid = torch.linspace(0.01, 0.5, 100).to(device)
    
    instr = Instrument(wavelength=1.54, I0=1e6)
    config = {
        'n_layers': 2,
        'instrument': instr,
        # Other ranges still needed
        'thickness_min': 50.0, 'thickness_max': 200.0,
        'roughness_min': 1.0, 'roughness_max': 5.0,
        'sld_min': 10.0, 'sld_max': 60.0,
        'apply_footprint': True
    }
    
    inputs, _ = generate_synthetic_batch(2, q_grid, config)
    assert inputs.shape == (2, 100)
    
    print("Generator with Instrument OK")

if __name__ == "__main__":
    test_structures()
    test_simulator_with_sample()
    test_generator_with_instrument()
