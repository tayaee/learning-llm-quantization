# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "aqlm",
#   "torch",
#   "transformers",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
# ]
# ///

"""AQLM 양자화 스크립트 (Additive Quantization, 2~3bit 벡터 양자화).

사용 예:
    uv run aqlm.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-aqlm
    uv run aqlm.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out --bits 2 --num-codebooks 1 --show-inference-instruction
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AQLM 초저비트 벡터 양자화 (2/3bit)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-AQLM-2bit-1x16")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # AQLM 논문 최적값: 2bit = 1x16 코드북, 3bit 근사도 유사. Llama-8B 기준.
    p.add_argument("--bits", type=int, default=2, choices=[2, 3])
    p.add_argument("--num-codebooks", type=int, default=1)
    p.add_argument("--nbits-per-codebook", type=int, default=16)
    p.add_argument("--in-group-size", type=int, default=8)
    p.add_argument("--out-group-size", type=int, default=1)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--val-size", type=int, default=1024)
    return p.parse_args()


def default_output_name(hf_model: str, bits: int, num_codebooks: int, nbits: int) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-AQLM-{bits}bit-{num_codebooks}x{nbits}"


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from aqlm import QuantizedLinear  # noqa: F401
        from aqlm.hf import quantize_model
    except ImportError as e:
        print(f"[aqlm][ERROR] aqlm 패키지 import 실패: {e}. 'uv run' 의존성이 설치됐는지 확인하세요.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[aqlm] 로드: {args.hf_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    print(f"[aqlm] 양자화: {args.bits}bit, {args.num_codebooks}x{args.nbits_per_codebook}, "
          f"in_group={args.in_group_size}")
    quantize_model(
        model, tokenizer,
        nsamples=args.nsamples, seqlen=args.seqlen, val_size=args.val_size,
        num_codebooks=args.num_codebooks, nbits_per_codebook=args.nbits_per_codebook,
        in_group_size=args.in_group_size, out_group_size=args.out_group_size,
        num_epochs=10 if args.bits == 2 else 5,
    )
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[aqlm] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(
        args.hf_model, args.bits, args.num_codebooks, args.nbits_per_codebook)
    write_readme(out_dir, model_name, args)
    print(f"[aqlm] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: AQLM {args.bits}bit ({args.num_codebooks}x{args.nbits_per_codebook})
---

# {model_name}

`{args.hf_model}` 의 **AQLM {args.bits}bit 가산 양자화** (2bit 1x16 코드북 기본).

## 기술 계보

- **이전: GPTQ/AWQ 스칼라 INT4** — 4bit 아래(3/2bit)에서는 스칼라 양자화 한계로 perplexity 폭증.
- **AQLM의 개선점**: 가중치 벡터를 여러 코드북 합으로 표현하는 가산 벡터양자화 + beam search 최적화로
  2bit에서도 Llama-8B 실용 품질 달성. PV-tuning(양자화+미세조정 병행)으로 추가 회복.
- **AQLM의 단점**: 전용 CUDA 커널 필요(aqlm/gptq CUDA), 양자화 시간 수 시간, vLLM 기본 미지원,
  코드북 오버헤드.
- **AQLM을 개선하는 후속 기법**: **QuIP#**(격자 코드북 + 비일관성 처리로 2bit SOTA),
  **AutoRound**(반올림 학습이 2bit에서도 경쟁), 경량 배포는 **GGUF Q2_K** 대안.

## 파라미터

- bits={args.bits}, num_codebooks={args.num_codebooks}, nbits_per_codebook={args.nbits_per_codebook},
  in_group_size={args.in_group_size}, out_group_size={args.out_group_size},
  nsamples={args.nsamples}, seqlen={args.seqlen}

## 사용법

```python
# pip install aqlm
from aqlm.hf import load_quantized_model
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("{model_name}")
m = load_quantized_model("{model_name}", device="cuda:0")
print(tok.decode(m.generate(tok("Hello", return_tensors="pt").input_ids.cuda(), max_new_tokens=64)[0]))
```
""",
        encoding="utf-8",
    )


def upload_to_hub(args: argparse.Namespace, output_dir: Path, model_name: str) -> str | None:
    if not args.upload:
        return None
    from huggingface_hub import HfApi
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[aqlm][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[aqlm][ERROR] whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add AQLM quantized model")
        print(f"[aqlm] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[aqlm][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ AQLM 추론 가이드 ({model_name}) ================
AQLM은 전용 커널이 필요합니다 (vLLM 기본 미지원).

    pip install aqlm torch transformers accelerate
    from aqlm.hf import load_quantized_model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}")
    m = load_quantized_model("{model_name}", device="cuda:0")
    ids = tok("Explain quantization in one sentence.", return_tensors="pt").input_ids.cuda()
    print(tok.decode(m.generate(ids, max_new_tokens=128)[0]))
================================================================
""")


def main() -> None:
    args = parse_args()
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(
        args.hf_model, args.bits, args.num_codebooks, args.nbits_per_codebook)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
