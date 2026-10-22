import torch

from xrefine.core.abeles import abeles


def check_gradients():
    # 학습 중 발생할 수 있는 극단적인 값 설정
    thick = torch.tensor([[1.0]], requires_grad=True)
    rough = torch.tensor([[2.0, 2.0]], requires_grad=True)
    sld = torch.tensor([[0.0, 20.0, 20.0]], requires_grad=True)
    q = torch.linspace(0.01, 0.5, 100).unsqueeze(0)

    # Forward
    r = abeles(q, thick, rough, sld)
    loss = r.sum()

    # Backward
    loss.backward()

    print(f"Thick Grad: {thick.grad.item():.3e}")
    print(f"Rough Grad: {rough.grad.abs().max().item():.3e}")
    print(f"SLD Grad:   {sld.grad.abs().max().item():.3e}")

    if torch.isnan(thick.grad) or thick.grad.abs() > 1e6:
        print("⚠️ Gradient is exploding or NaN!")


if __name__ == "__main__":
    check_gradients()
