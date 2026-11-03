"""Write a PEFT-format rank-1 LoRA adapter whose delta is -d d^T W (projects d out of W's output space).

Usage: make_rank1_lora.py <hf_model_dir> <out_dir> <tensor_name> [<tensor_name> ...] [--seed N]
tensor_name is the full HF weight name, e.g. model.language_model.layers.12.mlp.down_proj.weight
"""
import argparse, json, os
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

ap = argparse.ArgumentParser()
ap.add_argument("model_dir"); ap.add_argument("out_dir"); ap.add_argument("tensors", nargs="+")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
wmap = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
rng = np.random.default_rng(a.seed)
out, modules = {}, []
for name in a.tensors:
    with safe_open(os.path.join(a.model_dir, wmap[name]), "pt") as f:
        W = f.get_tensor(name).float().numpy()            # [out, in]
    d = rng.standard_normal(W.shape[0]).astype(np.float32); d /= np.linalg.norm(d)
    A = (d @ W)[None, :]                                   # [1, in]   = d^T W
    B = -d[:, None]                                        # [out, 1]
    base = "base_model.model." + name[: -len(".weight")]
    out[base + ".lora_A.weight"] = A.astype(np.float32)
    out[base + ".lora_B.weight"] = B.astype(np.float32)
    modules.append(name.split(".")[-2])
    print(f"{name}: W{W.shape} |dW|_F={np.linalg.norm(B @ A):.4f} |W|_F={np.linalg.norm(W):.4f}")
os.makedirs(a.out_dir, exist_ok=True)
save_file(out, os.path.join(a.out_dir, "adapter_model.safetensors"))
json.dump({"peft_type": "LORA", "r": 1, "lora_alpha": 1, "target_modules": sorted(set(modules)),
           "base_model_name_or_path": os.path.abspath(a.model_dir), "bias": "none", "task_type": "CAUSAL_LM",
           "fan_in_fan_out": False}, open(os.path.join(a.out_dir, "adapter_config.json"), "w"), indent=1)
