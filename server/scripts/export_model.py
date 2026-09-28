#!/usr/bin/env python3
"""Export the public DINOv2-small model to the ONNX file the server uses.

Downloads facebook/dinov2-small from Hugging Face (Apache-2.0), wraps it so the
output is the normalized CLS token (pooler_output), and writes an ONNX model
with fp16 weights and a fp32 input of 336x336.

Usage:
    pip install -r requirements-build.txt
    python scripts/export_model.py --out models/model.onnx

Input:  pixel_values  float32 [1, 3, 336, 336], ImageNet mean/std normalized.
Output: emb           float32 [1, 384].
"""
import argparse
import os

import torch
from transformers import AutoModel

MODEL_ID = "facebook/dinov2-small"
# 336 em vez dos 224 padrão: em cartas reais, detalhe fino decide entre prints
# parecidos (medido: top-1 16/19 → 18/19, margem +54%).
EMB_IN = 336


class Embedder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).pooler_output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models/model.onnx", help="output ONNX path")
    ap.add_argument("--fp32", action="store_true", help="keep fp32 weights (2x bigger file)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base = AutoModel.from_pretrained(MODEL_ID).eval()
    # O DINOv2 reamostra o position embedding (bicúbico, 37×37 → 24×24) a cada
    # forward. O Resize bicúbico reduzindo não roda no onnxruntime, então
    # pré-calcula em 336 e grava no modelo: no export o Resize vira escala 1.
    emb = base.embeddings
    n_patches = (EMB_IN // base.config.patch_size) ** 2
    with torch.no_grad():
        probe = torch.zeros(1, n_patches + 1, base.config.hidden_size)
        baked = emb.interpolate_pos_encoding(probe, EMB_IN, EMB_IN)
    emb.position_embeddings = torch.nn.Parameter(baked.clone())
    model = Embedder(base).eval()
    dummy = torch.zeros(1, 3, EMB_IN, EMB_IN)
    tmp = args.out + ".fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (dummy,), tmp, input_names=["pixel_values"], output_names=["emb"],
                          opset_version=17, dynamo=False)

    if args.fp32:
        os.replace(tmp, args.out)
    else:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        # keep_io_types: a entrada continua fp32, então o cliente não muda.
        # fp16 corta o arquivo pela metade sem perder acerto (int8 perdeu).
        # O conversor do onnxconverter_common deixava o Conv com tipos
        # misturados (fp32 × fp16) e o onnxruntime recusava carregar.
        m = convert_float_to_float16(onnx.load(tmp), keep_io_types=True)
        onnx.save(m, args.out)
        os.remove(tmp)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
