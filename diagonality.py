import argparse
from collections import defaultdict

import numpy as np
import torch

from conformer import Conformer, Attention


def diagonality(att):
    """
    att: torch.Tensor or np.ndarray with shape (B, NHeads, Tout, Tin)
    return: np.ndarray with shape (B, NHeads)
    """
    if isinstance(att, torch.Tensor):
        att = att.detach().cpu().numpy()

    bs, nheads, len_out, len_in = att.shape

    rel_distance = np.zeros((len_out, len_in), dtype=np.float32)
    for i in range(len_out):
        for j in range(len_in):
            rel_distance[i, j] = np.abs(i - j)

    denom = rel_distance.max(-1)
    denom[denom == 0] = 1.0

    result = (1.0 - (att * rel_distance).sum(-1) / denom).mean(-1)  # (B, NHeads)
    return result


def format_array(arr):
    return np.array2string(
        arr,
        formatter={"float_kind": lambda x: f"{x:.6f}"},
        separator=","
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze diagonality of Attention modules in custom Conformer."
    )
    parser.add_argument("--dim", type=int, required=True, help="Model dimension")
    parser.add_argument("--depth", type=int, required=True, help="Number of Conformer blocks")
    parser.add_argument("--heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--dim_head", type=int, required=True, help="Dimension per head")
    parser.add_argument("--seq_len", type=int, default=100, help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--n_batches", type=int, default=1, help="How many random batches to run")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    model = Conformer(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
    ).to(device)
    model.eval()

    outputs = {}
    handles = {}

    def hook_fn(module, inputs, output, name):
        # output from your Attention module is: (x, attn)
        if not isinstance(module, Attention):
            return

        if not isinstance(output, tuple) or len(output) != 2:
            print(f"[Warning] {name} output is not (x, attn). Skipping.")
            return

        _, attn = output

        if attn is None:
            print(f"[Warning] {name} attn is None. Skipping.")
            return

        outputs[name] = attn.detach().cpu()

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            handle = module.register_forward_hook(
                lambda module, inputs, output, name=name: hook_fn(module, inputs, output, name)
            )
            handles[name] = handle

    return_dict = defaultdict(list)

    with torch.no_grad():
        for batch_idx in range(args.n_batches):
            x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)

            try:
                _ = model(x)
            except Exception as e:
                print(f"[Warning] Forward failed at batch {batch_idx}: {e}")
                break

            if len(outputs) == 0:
                print("[Warning] No attention outputs were captured by hooks.")
                break

            for name, att in outputs.items():
                try:
                    diag = diagonality(att)  # (B, NHeads)
                    return_dict[name].append(diag)
                except Exception as e:
                    print(f"[Warning] Failed diagonality computation for {name}: {e}")

            outputs.clear()

    for _, handle in handles.items():
        handle.remove()

    lines = []

    if len(return_dict) == 0:
        lines.append("[Warning] No diagonality results collected.")
    else:
        for name, diags in return_dict.items():
            try:
                diags = np.concatenate(diags, axis=0)  # (num_samples, NHeads)
                line = (
                    f"{name}: "
                    f"nsamples={diags.shape[0]}, "
                    f"mean={format_array(diags.mean(0))}, "
                    f"std={format_array(diags.std(0))}"
                )
                lines.append(line)
            except Exception as e:
                lines.append(f"[Warning] Failed to summarize {name}: {e}")

    text = "\n".join(lines)
    print(text)

    if args.log_file is not None:
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()