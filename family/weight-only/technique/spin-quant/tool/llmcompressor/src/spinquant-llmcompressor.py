# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "llmcompressor",
#   "torch",
#   "transformers",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///

"""SpinQuant 양자화 스크립트 — llmcompressor 구현 (SpinQuant-style Hadamard 변환).

공식 예제 레시피: SpinQuantModifier(rotations, hadamard) +
QuantizationModifier(W4A16) → oneshot(pipeline="datafree").
캘리브레이션 데이터가 필요 없는 datafree 파이프라인이다.

사용 예:
    uv run spinquant-llmcompressor.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-sq-lc
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpinQuant-style 양자화 (llmcompressor, datafree)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-SpinQuant-W4A16")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # 공식 spinquant_example.py 기준 최적값
    p.add_argument("--rotations", default="R1,R2,R4",
                   help="적용할 회전 (콤마 구분, 공식 예제: R1,R2,R4)")
    p.add_argument("--transform-block-size", type=int, default=128,
                   help="Hadamard 블록 크기 (양자화 group 크기와 일치 권장)")
    p.add_argument("--transform-type", default="hadamard", choices=["hadamard", "random"])
    p.add_argument("--scheme", default="W4A16", help="W4A16이 SpinQuant 공식 예제 표준")
    return p.parse_args()


def default_output_name(hf_model: str, scheme: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-SpinQuant-{scheme}"


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as e:
        print(f"[spinquant-lc][ERROR] llmcompressor import 실패: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        from llmcompressor.modifiers.transform import SpinQuantModifier
    except ImportError as e:
        print(f"[spinquant-lc][ERROR] SpinQuantModifier import 실패: {e} "
              f"(llmcompressor>=0.7 필요)", file=sys.stderr)
        sys.exit(1)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[spinquant-lc] 로드: {args.hf_model} (datafree, 회전={args.rotations})")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    rotations = [r.strip() for r in args.rotations.split(",") if r.strip()]
    recipe = [
        SpinQuantModifier(rotations=rotations,
                          transform_block_size=args.transform_block_size,
                          transform_type=args.transform_type),
        QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=["lm_head"]),
    ]
    oneshot(model=model, recipe=recipe, pipeline="datafree")

    model.save_pretrained(str(out_dir), save_compressed=True)
    tokenizer.save_pretrained(str(out_dir))
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[spinquant-lc] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model, args.scheme)
    write_readme(out_dir, model_name, args)
    print(f"[spinquant-lc] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: SpinQuant-style {args.scheme} via llmcompressor (datafree)
---

# {model_name}

`{args.hf_model}` 의 **SpinQuant-style {args.scheme}** 양자화 (llmcompressor, 무캘리브).

## 기술 계보 (도구 관점)

- **이전: Meta 공식 SpinQuant 학습 파이프라인** — Cayley SGD로 회전을 학습해
  최고 품질을 내지만 8B 수십 분~수 시간 + 별도 저장소·스크립트 필요.
- **본 도구의 개선점**: 고정 Hadamard 회전(R1/R2/R4, block={args.transform_block_size})을
  datafree로 융합 → 캘리브레이션 없이 수 분 내 `compressed-tensors` 산출,
  vLLM 무변환 연동.
- **단점**: 학습된 회전이 아니라 고정 회전이라 W4A4 극저비트에서는
  공식 학습 파이프라인(`spin-quant/tool/meta/`)보다 품질이 낮다.
- **대안**: 최고 품질은 `tool/meta/`, 경량 자작 구현은 `tool/python/`.

## 파라미터

- rotations={args.rotations}, transform_block_size={args.transform_block_size},
  transform_type={args.transform_type}, scheme={args.scheme}

## 사용법

```bash
vllm serve {model_name} --quantization compressed-tensors --max-model-len 8192
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
        print("[spinquant-lc][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[spinquant-lc][ERROR] whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id,
                          commit_message="Add SpinQuant quantized model (llmcompressor)")
        print(f"[spinquant-lc] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[spinquant-lc][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ SpinQuant(llmcompressor) 추론 가이드 ({model_name}) ================
[1] vLLM 서빙 (compressed-tensors):
    vllm serve {model_name} --quantization compressed-tensors --tensor-parallel-size 1 --max-model-len 8192

[2] transformers:
    pip install llmcompressor accelerate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}")
    m = AutoModelForCausalLM.from_pretrained("{model_name}", device_map="auto", torch_dtype="auto")
    print(tok.decode(m.generate(tok("Hello, my name is", return_tensors="pt").input_ids.cuda(), max_new_tokens=128)[0]))
================================================================
""")


def main() -> None:
    args = parse_args()
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(args.hf_model, args.scheme)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
