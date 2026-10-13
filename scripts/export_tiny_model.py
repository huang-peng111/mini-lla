"""生成阶段 3 使用的 Tiny F32 模型，不依赖第三方 Python 包。"""
import json
import os
import random
import struct

def main():
    # 使用固定随机种子，保证教学模型可以重复生成和复现测试结果。
    output = os.path.join("models", "tiny")
    os.makedirs(output, exist_ok=True)
    config = {"vocab_size": 128, "dim": 32, "hidden_dim": 64,
              "n_layers": 2, "n_heads": 4, "n_kv_heads": 4,
              "max_seq_len": 128, "rope_theta": 10000.0,
              "rms_norm_eps": 1e-5}
    rng = random.Random(42)
    tensors = []
    def add(name, shape, values):
        tensors.append({"name": name, "shape": list(shape), "data": values})
    def random_values(count):
        return [rng.gauss(0.0, 0.02) for _ in range(count)]
    dim, hidden, vocab = config["dim"], config["hidden_dim"], config["vocab_size"]
    add("token_embedding", (vocab, dim), random_values(vocab * dim))
    for layer in range(config["n_layers"]):
        prefix = f"layers.{layer}."
        add(prefix + "attention_norm", (dim,), [1.0] * dim)
        for name, shape in [("wq", (dim, dim)), ("wk", (dim, dim)), ("wv", (dim, dim)), ("wo", (dim, dim)), ("w_gate", (hidden, dim)), ("w_up", (hidden, dim)), ("w_down", (dim, hidden))]:
            add(prefix + name, shape, random_values(shape[0] * shape[1]))
        add(prefix + "ffn_norm", (dim,), [1.0] * dim)
    add("final_norm", (dim,), [1.0] * dim)
    add("lm_head", (vocab, dim), random_values(vocab * dim))
    offset, metadata = 0, []
    # model.bin 与 model.json 使用相同的 tensor 顺序和 offset。
    with open(os.path.join(output, "model.bin"), "wb") as binary:
        for tensor in tensors:
            binary.write(struct.pack("<" + "f" * len(tensor["data"]), *tensor["data"]))
            size = len(tensor["data"]) * 4
            metadata.append({"name": tensor["name"], "shape": tensor["shape"], "dtype": "float32", "offset": offset, "byte_size": size})
            offset += size
    manifest = {"config": config, "tokenizer": {"type": "ascii", "bos_id": 1, "eos_id": 2, "unk_id": 0}, "tensors": metadata}
    with open(os.path.join(output, "model.json"), "w", encoding="utf-8") as text:
        json.dump(manifest, text, indent=2)
    print(f"wrote {output}")

if __name__ == "__main__":
    main()
