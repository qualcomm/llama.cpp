import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gguf-py"))
import importlib
GGUFReader = importlib.import_module("gguf.gguf_reader").GGUFReader

KEYS = [
    "block_count", "context_length", "embedding_length", "feed_forward_length",
    "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.value_length",
    "attention.sliding_window", "attention.sliding_window_pattern",
    "rope.dimension_count", "rope.freq_base",
    "expert_count", "expert_used_count", "expert_feed_forward_length",
    "expert_shared_count", "expert_shared_feed_forward_length",
    "attention.scale", "attn_logit_softcapping", "final_logit_softcapping",
    # ssm / linear-attn / gated delta net
    "ssm.conv_kernel", "ssm.inner_size", "ssm.state_size",
    "ssm.time_step_rank", "ssm.group_count", "ssm.head_dim",
    "linear_attn", "n_lora",
]

def val(r, k):
    f = r.fields.get(k)
    if f is None: return None
    try:
        return f.contents()
    except Exception:
        try:
            return f.parts[f.data[0]].tolist()
        except Exception:
            return "?"

def dump(path):
    r = GGUFReader(path)
    arch = val(r, "general.architecture")
    name = val(r, "general.name")
    out = {"file": os.path.basename(path), "arch": arch, "name": name}
    # collect all keys that exist for any arch prefix
    for k in KEYS:
        for fk in r.fields:
            if fk == f"{arch}.{k}" or fk == f"general.{k}":
                out[k] = val(r, fk)
                break
    # also scan for any per-layer / hybrid type keys
    for fk in list(r.fields):
        if any(t in fk for t in (".attention.layer", "rope.scaling", "swa", "is_swa", "recurrent")):
            out[fk] = val(r, fk)
    return out

files = sys.argv[1:]
for p in files:
    if not os.path.exists(p):
        print("MISSING:", p); continue
    try:
        d = dump(p)
    except Exception as e:
        print("ERR", p, e); continue
    print("="*70)
    print(d.get("file"))
    print("  arch:", d.get("arch"), "| name:", d.get("name"))
    for k in d:
        if k in ("file","arch","name"): continue
        print(f"  {k}: {d[k]}")
