"""Extract tensor state from a trusted legacy CFNO adapter, without running it.

Old dill embeds references to unavailable kernel helper functions. Placeholders
are restricted to legacy mamba/causal-conv imports in this extraction process;
none are executed or included in the tensor-only output used for training.
"""
import argparse
import json
from pathlib import Path
import sys
import types

import dill
import torch
import mamba_ssm.utils as utils


def forbidden(*args, **kwargs):
    raise RuntimeError("Legacy extraction placeholder must never execute")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).parent / "cfno_upstream"))
    sys.modules["mamba_ssm.utils.torch"] = torch
    utils.torch = torch
    missing = []

    class Extractor(dill.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                if module.startswith(("mamba_ssm.", "causal_conv1d.")):
                    missing.append(module + "." + name)
                    return forbidden
                raise

    pm = types.ModuleType("dill")
    pm.__dict__.update(vars(dill))
    pm.Unpickler = Extractor
    original = torch.load(args.source, map_location="cpu", pickle_module=pm,
                          weights_only=False, mmap=True)
    state = {k: v.detach().cpu().clone() for k, v in original.state_dict().items()
             if torch.is_tensor(v) and "grid" not in k}
    assert all(torch.isfinite(v).all() for v in state.values())
    assert "post_lift_ssm.ssm.A_log" in state
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    destination.with_suffix(".json").write_text(json.dumps({"source": args.source,
        "extraction_only_unavailable_functions": missing,
        "tensors": {k: list(v.shape) for k, v in state.items()}}, indent=2))


if __name__ == "__main__":
    main()
