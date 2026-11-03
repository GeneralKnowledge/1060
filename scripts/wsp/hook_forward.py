"""Load Qwen3.5-0.8B fp32 in transformers on GPU, run one forward with a hook on layer 15 MLP output."""
import json, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = sys.argv[1]
t0 = time.time()
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to("cuda").eval()
torch.cuda.synchronize(); load_s = time.time() - t0
caught = {}
layer = model.model.layers[15]
h = layer.mlp.register_forward_hook(lambda m, i, o: caught.__setitem__("mlp15", o.detach()))
text = open(sys.argv[2]).read()[:20000]
ids = tok(text, return_tensors="pt").input_ids[:, :512].to("cuda")
res = {"class": type(model).__name__, "layer15_type": type(layer).__name__, "load_s": round(load_s, 1),
       "vram_after_load_mib": round(torch.cuda.memory_allocated() / 2**20)}
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    for k in range(3):
        torch.cuda.synchronize(); t = time.time(); out = model(ids); torch.cuda.synchronize()
        res[f"forward{k}_s"] = round(time.time() - t, 3)
res.update({"tokens": ids.shape[1], "hook_shape": list(caught["mlp15"].shape), "hook_norm": round(caught["mlp15"].norm().item(), 3),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "logits_finite": bool(torch.isfinite(out.logits).all())})
print(json.dumps(res))
