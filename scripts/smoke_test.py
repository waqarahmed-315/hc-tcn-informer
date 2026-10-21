"""CPU smoke test for standard initialization only (no ETT data required)."""
import torch

from models.informer import Informer
from models.tcn_informer import TCNInformer


def main():
    torch.manual_seed(17)
    for cls in (Informer, TCNInformer):
        model = cls(
            enc_in=7, dec_in=7, c_out=1,
            seq_len=48, label_len=24, out_len=12,
            d_model=32, n_heads=4, d_ff=64, e_layers=1, d_layers=1,
            attn="full", tcn_channels=[16, 32] if cls is TCNInformer else None,
            init_mode="standard", use_revin=True, device=torch.device("cpu"),
        ) if cls is TCNInformer else cls(
            enc_in=7, dec_in=7, c_out=1, seq_len=48, label_len=24,
            out_len=12, d_model=32, n_heads=4, d_ff=64, e_layers=1, d_layers=1,
            attn="full", init_mode="standard", use_revin=True,
            device=torch.device("cpu"),
        )
        model.eval()
        b = 2
        with torch.no_grad():
            y = model(torch.randn(b,48,7), torch.randn(b,48,4),
                      torch.randn(b,36,7), torch.randn(b,36,4))
        assert y.shape == (b,12,1), (cls.__name__,y.shape)
        assert torch.isfinite(y).all(), cls.__name__
        print(f"{cls.__name__}: forward pass OK, output shape {tuple(y.shape)}")


if __name__ == "__main__":
    main()
