import torch

from xrefine.core.interface import Simulator
from xrefine.logger import logger

def test_forward_pass():
    logger.info("Testing Forward Pass...")
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
    logger.info("Forward Pass: OK")
    return q, r

def test_gradients():
    logger.info("\nTesting Gradients (PINN capability)...")
    sim = Simulator(method='abeles', device='cpu')
    
    thickness = torch.tensor([[100.0]], requires_grad=True)
    roughness = torch.tensor([[2.0, 2.0]], requires_grad=True)
    sld = torch.tensor([[0.0, 20.0, 20.0]], requires_grad=True)
    q = torch.linspace(0.01, 0.2, 50).unsqueeze(0)
    
    r = sim(q, thickness, roughness, sld)
    loss = r.sum()
    loss.backward()
    
    logger.info(f"Thickness Grad:\n{thickness.grad}")
    logger.info(f"Roughness Grad:\n{roughness.grad}")
    logger.info(f"SLD Grad:\n{sld.grad}")
    
    # Check gradients are not None and not zero (unless physically so)
    assert thickness.grad is not None
    assert roughness.grad is not None
    assert sld.grad is not None
    
    # Check physics: thicker layer usually changes oscillation frequency, so grad should be non-zero
    assert torch.abs(thickness.grad).sum() > 0
    
    logger.info("Gradients: OK")

def test_smearing():
    logger.info("\nTesting Smearing...")
    sim = Simulator(method='abeles', device='cpu')
    
    dq = torch.tensor([[0.01]]) # constant resolution
    q = torch.linspace(0.01, 0.5, 100).unsqueeze(0)
    
    thickness = torch.tensor([[300.0]])
    roughness = torch.tensor([[2.0, 2.0]])
    sld = torch.tensor([[0.0, 20.0, 20.0]])
    
    r_smeared = sim(q, thickness, roughness, sld, dq=dq)
    
    assert r_smeared.shape == q.shape
    logger.info("Smearing: OK")

def test_kinematical():
    logger.info("\nTesting Kinematical...")
    sim = Simulator(method='kinematical', device='cpu')
    
    q = torch.linspace(0.01, 0.5, 100).unsqueeze(0)
    thickness = torch.tensor([[100.0]])
    roughness = torch.tensor([[2.0, 2.0]])
    sld = torch.tensor([[0.0, 10.0, 20.0]]) # Amb, Layer, Sub
    
    r = sim(q, thickness, roughness, sld)
    assert r.shape == q.shape
    logger.info("Kinematical: OK")

if __name__ == "__main__":
    try:
        test_forward_pass()
        test_gradients()
        test_smearing()
        test_kinematical()
        logger.info("\nALL TESTS PASSED!")
    except Exception as e:
        logger.info(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
