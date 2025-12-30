import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from xrefine.simulator.interface import Simulator
from xrefine.simulator.abeles import abeles
from xrefine.simulator.kinematical import kinematical_approximation

def test_forward_pass():
    print("Testing Forward Pass...")
    sim = Simulator(method='abeles', device='cpu')
    
    batch_size = 2
    q = torch.linspace(0.01, 0.5, 100).unsqueeze(0).repeat(batch_size, 1)
    
    # 2 layers: [Ambient, L1, L2, Substrate] -> SLD has 4 values
    # Thickness: [L1, L2] -> 2 values
    # Roughness: [Amb/L1, L1/L2, L2/Sub] -> 3 values
    
    thickness = torch.tensor([[100.0, 50.0], [200.0, 20.0]])
    roughness = torch.tensor([[2.0, 3.0, 2.0], [5.0, 5.0, 5.0]])
    sld = torch.tensor([[0.0, 20.0, 10.0, 20.0], [0.0, 15.0, 15.0, 20.0]])
    
    # Run simulation
    r = sim(q, thickness, roughness, sld)
    
    assert r.shape == q.shape
    assert torch.all(r >= 0) and torch.all(r <= 1.0)
    print("Forward Pass: OK")
    return q, r

def test_gradients():
    print("\nTesting Gradients (PINN capability)...")
    sim = Simulator(method='abeles', device='cpu')
    
    thickness = torch.tensor([[100.0]], requires_grad=True)
    roughness = torch.tensor([[2.0, 2.0]], requires_grad=True)
    sld = torch.tensor([[0.0, 20.0, 20.0]], requires_grad=True)
    q = torch.linspace(0.01, 0.2, 50).unsqueeze(0)
    
    r = sim(q, thickness, roughness, sld)
    loss = r.sum()
    loss.backward()
    
    print(f"Thickness Grad:\n{thickness.grad}")
    print(f"Roughness Grad:\n{roughness.grad}")
    print(f"SLD Grad:\n{sld.grad}")
    
    # Check gradients are not None and not zero (unless physically so)
    assert thickness.grad is not None
    assert roughness.grad is not None
    assert sld.grad is not None
    
    # Check physics: thicker layer usually changes oscillation frequency, so grad should be non-zero
    assert torch.abs(thickness.grad).sum() > 0
    
    print("Gradients: OK")

def test_smearing():
    print("\nTesting Smearing...")
    sim = Simulator(method='abeles', device='cpu')
    
    dq = torch.tensor([[0.01]]) # constant resolution
    q = torch.linspace(0.01, 0.5, 100).unsqueeze(0)
    
    thickness = torch.tensor([[300.0]])
    roughness = torch.tensor([[2.0, 2.0]])
    sld = torch.tensor([[0.0, 20.0, 20.0]])
    
    r_smeared = sim(q, thickness, roughness, sld, dq=dq)
    
    assert r_smeared.shape == q.shape
    print("Smearing: OK")

def test_kinematical():
    print("\nTesting Kinematical...")
    sim = Simulator(method='kinematical', device='cpu')
    
    q = torch.linspace(0.01, 0.5, 100).unsqueeze(0)
    thickness = torch.tensor([[100.0]])
    roughness = torch.tensor([[2.0, 2.0]])
    sld = torch.tensor([[0.0, 10.0, 20.0]]) # Amb, Layer, Sub
    
    r = sim(q, thickness, roughness, sld)
    assert r.shape == q.shape
    print("Kinematical: OK")

if __name__ == "__main__":
    try:
        test_forward_pass()
        test_gradients()
        test_smearing()
        test_kinematical()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
