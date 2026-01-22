"""
Unit Tests for Enhanced Gradient Flow
======================================
Tests to verify soft clamping preserves gradients at boundaries.
"""
import torch
from xrefine.physics_utils_enhanced import (
    soft_clamp, 
    soft_clamp_min, 
    curriculum_weight,
    physics_constraint_penalty
)


def test_gradient_preservation():
    """Test that soft clamp preserves gradients at boundaries"""
    print("\n" + "="*80)
    print("TEST 1: Gradient Preservation at Boundaries")
    print("="*80)
    
    # Test at upper boundary (x = 2.0, clamp to [-1, 1])
    x = torch.tensor([2.0], requires_grad=True)
    
    # Hard clamp
    y_hard = torch.clamp(x, -1.0, 1.0)
    y_hard.backward()
    grad_hard = x.grad.item()
    
    # Soft clamp
    x.grad = None
    x = torch.tensor([2.0], requires_grad=True)
    y_soft = soft_clamp(x, -1.0, 1.0, sharpness=10.0)
    y_soft.backward()
    grad_soft = x.grad.item()
    
    print(f"Input: x = 2.0 (beyond upper bound of 1.0)")
    print(f"Hard clamp gradient: {grad_hard:.6f} (gradient blocked!)")
    print(f"Soft clamp gradient: {grad_soft:.6f} (gradient preserved!)")
    
    assert grad_hard == 0.0, "Hard clamp should block gradients"
    assert grad_soft != 0.0, "Soft clamp should preserve gradients"
    assert abs(grad_soft) > 1e-6, "Soft clamp gradient should be non-negligible"
    
    print("✅ PASSED: Soft clamp preserves gradients at boundaries\n")
    return True


def test_soft_clamp_values():
    """Test that soft clamp produces reasonable values"""
    print("="*80)
    print("TEST 2: Soft Clamp Value Range")
    print("="*80)
    
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = soft_clamp(x, -1.0, 1.0, sharpness=10.0)
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
    
    # Check bounds (soft, so not exact)
    assert y.min() > -1.1, "Soft clamp should approximately respect lower bound"
    assert y.max() < 1.1, "Soft clamp should approximately respect upper bound"
    
    # Check that values inside range are approximately preserved
    mid_val = y[2].item()  # x=0 should map to ~0
    assert abs(mid_val) < 0.1, f"Middle value should be near 0, got {mid_val}"
    
    print("✅ PASSED: Soft clamp produces reasonable values\n")
    return True


def test_curriculum_weight_schedule():
    """Test curriculum learning weight schedule"""
    print("="*80)
    print("TEST 3: Curriculum Learning Weight Schedule")
    print("="*80)
    
    max_epochs = 1000
    test_epochs = [0, 150, 300, 500, 999]
    
    print(f"Schedule (warmup_fraction=0.3, max_epochs={max_epochs}):")
    for epoch in test_epochs:
        weight = curriculum_weight(epoch, max_epochs)
        print(f"  Epoch {epoch:4d}: weight = {weight:.3f}")
    
    # Check start and end
    start_weight = curriculum_weight(0, max_epochs)
    end_weight = curriculum_weight(999, max_epochs)
    
    assert abs(start_weight - 0.1) < 0.01, "Should start at 0.1"
    assert abs(end_weight - 1.0) < 0.01, "Should end at 1.0"
    
    # Check monotonic increase
    prev_weight = 0.0
    for epoch in range(0, max_epochs, 50):
        weight = curriculum_weight(epoch, max_epochs)
        assert weight >= prev_weight, "Weight should be monotonically increasing"
        prev_weight = weight
    
    print("✅ PASSED: Curriculum weight schedule is correct\n")
    return True


def test_physics_constraint_penalty():
    """Test physics constraint penalty calculation"""
    print("="*80)
    print("TEST 4: Physics Constraint Penalty")
    print("="*80)
    
    # Valid case: roughness < 0.5 * thickness
    thickness = torch.tensor([[100.0, 200.0]])
    roughness = torch.tensor([[30.0, 80.0]])  # Both valid
    
    penalty_valid = physics_constraint_penalty(thickness, roughness)
    print(f"Valid case (roughness < 0.5 * thickness): penalty = {penalty_valid.item():.6f}")
    assert penalty_valid.item() < 1e-6, "Valid configuration should have near-zero penalty"
    
    # Invalid case: roughness > 0.5 * thickness
    roughness_invalid = torch.tensor([[60.0, 120.0]])  # Both violate constraint
    penalty_invalid = physics_constraint_penalty(thickness, roughness_invalid)
    print(f"Invalid case (roughness > 0.5 * thickness): penalty = {penalty_invalid.item():.6f}")
    assert penalty_invalid.item() > 0.0, "Invalid configuration should have positive penalty"
    
    print("✅ PASSED: Physics constraint penalty works correctly\n")
    return True


def test_soft_clamp_min():
    """Test soft lower bound clamping"""
    print("="*80)
    print("TEST 5: Soft Lower Bound Clamping")
    print("="*80)
    
    x = torch.tensor([1e-20, 1e-15, 1e-10, 1e-5, 1.0], requires_grad=True)
    y = soft_clamp_min(x, min_val=1e-15, sharpness=10.0)
    
    print(f"Input:  {[f'{v:.2e}' for v in x.tolist()]}")
    print(f"Output: {[f'{v:.2e}' for v in y.tolist()]}")
    
    # All values should be >= min_val (approximately)
    assert (y >= 1e-16).all(), "All values should respect lower bound"
    
    # Check gradient preservation
    y.sum().backward()
    assert x.grad is not None, "Gradients should flow"
    assert (x.grad != 0).any(), "At least some gradients should be non-zero"
    
    print("✅ PASSED: Soft lower bound preserves gradients\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 " * 30)
    print("ENHANCED GRADIENT FLOW TESTS")
    print("🧪 " * 30)
    
    tests = [
        test_gradient_preservation,
        test_soft_clamp_values,
        test_curriculum_weight_schedule,
        test_physics_constraint_penalty,
        test_soft_clamp_min
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
            failed += 1
    
    print("="*80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("🎉 All tests passed successfully!")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
