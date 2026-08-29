#!/usr/bin/env python3
"""Split the routed experts of a MoE GGUF into a hot and a cold tensor (SPEC.md 4.1).

Each `*_exps` tensor becomes two tensors:
  <name>_hot   experts [0, H)         followed by n_expert_used zeroed mask slots
  <name>_cold  experts [H, n_expert)  followed by n_expert_used zeroed mask slots
`ffn_gate_inp` rows are permuted the same way, so the router emits the new numbering
directly and no runtime remap of the ids is needed.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, "/home/tim/projects/a/llama.cpp/gguf-py")
from gguf import GGUFReader, GGUFWriter, GGUFEndian
from gguf.constants import GGMLQuantizationType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="source gguf, or all shards in order")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--hot-set", required=True, help="npy int array [n_layer, H] of expert ids")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--merge-gate-up", action="store_true",
                    help="emit one gate_up tensor per branch, halving the expert matmul count")
    a = ap.parse_args()

    hot = np.load(a.hot_set)
    n_layer_hot, H = hot.shape
    print("hot set: %d layers x %d experts" % (n_layer_hot, H), flush=True)

    readers = [GGUFReader(p) for p in a.src]
    r = readers[0]
    arch = a.arch or bytes(r.fields["general.architecture"].parts[-1]).decode()
    n_expert = int(r.fields[arch + ".expert_count"].parts[-1][0])
    n_used   = int(r.fields[arch + ".expert_used_count"].parts[-1][0])
    n_cold   = n_expert - H
    print("arch=%s n_expert=%d n_used=%d H=%d n_cold=%d" % (arch, n_expert, n_used, H, n_cold), flush=True)

    w = GGUFWriter(a.dst, arch, endianess=GGUFEndian.LITTLE)

    skip = {"general.architecture", "split.no", "split.count", "split.tensors.count"}
    for key, field in r.fields.items():
        if key in skip or not field.types:
            continue
        w.add_key_value(key, field.contents(), field.types[0], sub_type=field.types[-1] if len(field.types) > 1 else None)
    w.add_uint32(arch + ".expert_hot_count", H)

    def layer_of(name):
        parts = name.split(".")
        return int(parts[1]) if len(parts) > 2 and parts[0] == "blk" else None

    all_tensors = [t for rd in readers for t in rd.tensors]
    by_name = {str(t.name): t for t in all_tensors}
    print("tensors: %d across %d file(s)" % (len(all_tensors), len(readers)), flush=True)

    for t in all_tensors:
        name = str(t.name)

        if a.merge_gate_up and name.endswith("ffn_up_exps.weight"):
            continue  # folded into the gate_up tensor below
        il = layer_of(name)
        raw = np.asarray(t.data)

        merge = a.merge_gate_up and name.endswith("ffn_gate_exps.weight")
        if merge:
            up = by_name.get(name.replace("ffn_gate_exps", "ffn_up_exps"))
            if up is None or up.tensor_type != t.tensor_type:
                merge = False

        if name.endswith("_exps.weight") and il is not None and il < n_layer_hot:
            order = hot[il]
            assert len(set(order.tolist())) == H, "hot set for layer %d has duplicates" % il
            cold_ids = np.array(sorted(set(range(n_expert)) - set(order.tolist())), dtype=np.int64)
            assert len(cold_ids) == n_cold

            # raw is the byte layout with the expert as the outermost axis
            assert raw.shape[0] == n_expert, (name, raw.shape)
            base = name[:-len(".weight")]
            if merge:
                # gate rows first, then up rows, matching the merged layout the graph expects
                raw = np.concatenate([raw, np.asarray(up.data)], axis=1)
                base = base.replace("ffn_gate_exps", "ffn_gate_up_exps")

            zeros = np.zeros((n_used,) + raw.shape[1:], dtype=raw.dtype)

            hot_data  = np.concatenate([raw[order],    zeros], axis=0)
            cold_data = np.concatenate([raw[cold_ids], zeros], axis=0)

            w.add_tensor(base + "_hot.weight",  hot_data,  raw_dtype=t.tensor_type)
            w.add_tensor(base + "_cold.weight", cold_data, raw_dtype=t.tensor_type)
            print("  %-32s -> %s hot[%d] cold[%d]" % (name, base.split(".")[-1], H + n_used, n_cold + n_used), flush=True)
            continue

        if name.endswith("ffn_gate_inp.weight") and il is not None and il < n_layer_hot:
            order = hot[il]
            cold_ids = np.array(sorted(set(range(n_expert)) - set(order.tolist())), dtype=np.int64)
            perm = np.concatenate([order, cold_ids])
            assert raw.shape[0] == n_expert, (name, raw.shape)
            w.add_tensor(name, raw[perm], raw_dtype=t.tensor_type)
            continue

        w.add_tensor(name, raw, raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print("written", a.dst, "%.2f GiB" % (os.path.getsize(a.dst)/1024**3), flush=True)


if __name__ == "__main__":
    main()
