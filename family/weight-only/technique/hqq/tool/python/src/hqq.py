# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "hqq",
#   "torch",
#   "transformers",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///

"""HQQ 양자화 스크립트 (Half-Quadratic Quantization, 무캘리브 초고속).

사용 예:
    uv run hqq.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-hqq
    uv run hqq.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out --bits 4 --group-size 64 --show-inference-instruction
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HQQ 무캘리브 고속 양자화")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-HQQ-Int4-64g")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # HQQ 최적 기본값: Llama-8B는 group 64 + axis 0 + optimize가 품질 최적
    p.add_argument("--bits", type=int, default=4, choices=[1, 2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--axis", type=int, default=0)
    p.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attn-bits", type=int, default=None, help="어텐션 별도 비트 (미지정 시 --bits 따름)")
    p.add_argument("--mlp-bits", type=int, default=None)
    return p.parse_args()


def default_output_name(hf_model: str, bits: int, group_size: int) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-HQQ-Int{bits}-{group_size}g"


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
    except ImportError as e:
        print(f"[hqq][ERROR] hqq import 실패: {e}", file=sys.stderr)
        sys.exit(1)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hqq] 로드: {args.hf_model} (무캘리브, half-quadratic 최적화)")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    attn_cfg = BaseQuantizeConfig(nbits=args.attn_bits or args.bits, group_size=args.group_size,
                                  quant_zero=not args.optimize, quant_scale=False, axis=args.axis)
    mlp_cfg = BaseQuantizeConfig(nbits=args.mlp_bits or args.bits, group_size=args.group_size,
                                 quant_zero=not args.optimize, quant_scale=False, axis=args.axis)
    quant_config = {"nbits": args.bits, "group_size": args.group_size}

    # Linear 층을 HQQLinear로 교체
    for name, mod in list(model.named_modules()):
        if mod.__class__.__name__ != "Linear":
            continue
        cfg = attn_cfg if "self_attn" in name else mlp_cfg
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        hqq_lin = HQQLinear(mod, quant_config=cfg, compute_dtype=torch.bfloat16, device=mod.weight.device)
        setattr(parent, leaf, hqq_lin)
    # 연속 완화(half-quadratic) 최적화는 HQQLinear 생성 시 자동 수행됨 (optimize 플래그 반영)

    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "hqq_config.json").write_text(json.dumps(
        {"bits": args.bits, "group_size": args.group_size, "axis": args.axis,
         "optimize": args.optimize, "attn_bits": args.attn_bits, "mlp_bits": args.mlp_bits,
         "quant_config": quant_config}, indent=2), encoding="utf-8")
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[hqq] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model, args.bits, args.group_size)
    write_readme(out_dir, model_name, args)
    print(f"[hqq] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: HQQ INT{args.bits} (group={args.group_size}, axis={args.axis})
---

# {model_name}

`{args.hf_model}` 의 **HQQ INT{args.bits}** 양자화 — 캘리브레이션 데이터 없이 수 분 내 완료.

## 기술 계보

- **이전: GPTQ/AWQ** — 캘리브레이션(128~512 샘플, 수십 분)과 데이터 편향 의존이 필수.
- **HQQ의 개선점**: Half-Quadratic 분해로 스케일/제로점을 닫힌 형태+교대 최적화로 탐색,
  무캘리브·초고속(8B 수 분)임에도 INT4 품질이 GPTQ-128g에 근접. 1~2bit 극저비트도 지원.
- **HQQ의 단점**: 전용 HQQLinear 커널 의존, vLLM 통합은 llmcompressor 경유,
  활성화 양자화 미포함(W4A16).
- **HQQ를 개선하는 후속 기법**: **AutoRound**(반올림 학습으로 품질 상한+),
  **QuaRot/SpinQuant**(회전 추가로 활성화 양자화까지), **FP8**(Blackwell 네이티브 속도).

## 파라미터

- bits={args.bits}, group_size={args.group_size}, axis={args.axis}, optimize={args.optimize}

## 사용법

```bash
vllm serve {model_name} --quantization compressed-tensors --max-model-len 8192
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('{model_name}', device_map='auto')"
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
        print("[hqq][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        print("[hqq][ERROR] 업로드 계정 미지정. --hf-user 또는 HF_USER 환경변수로 명시하세요.",
              file=sys.stderr)
        sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add HQQ quantized model")
        print(f"[hqq] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[hqq][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ HQQ 추론 가이드 ({model_name}) ================
[1] vLLM 서빙 (compressed-tensors):
    vllm serve {model_name} --quantization compressed-tensors --tensor-parallel-size 1 --max-model-len 8192

[2] transformers (+hqq):
    pip install hqq accelerate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}")
    m = AutoModelForCausalLM.from_pretrained("{model_name}", device_map="auto", torch_dtype="auto", trust_remote_code=True)
    print(tok.decode(m.generate(tok("Hello, my name is", return_tensors="pt").input_ids.cuda(), max_new_tokens=128)[0]))
================================================================
""")


def main() -> None:
    args = parse_args()
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(args.hf_model, args.bits, args.group_size)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
