# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "auto-round",
#   "torch",
#   "transformers",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
# ]
# ///

"""AutoRound 양자화 스크립트 (Intel, 반올림 최적화 기반 INT4/INT2).

사용 예:
    uv run auto-round.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-autoround
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intel AutoRound 반올림 최적화 양자화")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-AutoRound-Int4-128g")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # AutoRound 논문/레시피 최적값 (Llama-3.1 8B, DGX Spark)
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--iters", type=int, default=200, help="반올림 최적화 반복 (200이 품질/시간 균형)")
    p.add_argument("--dataset", default="NeelNanda/pile-10k", help="기본 pile-10k (AutoRound 공식 레시피)")
    return p.parse_args()


def default_output_name(hf_model: str, bits: int) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-AutoRound-Int{bits}-128g"


def run_quantization(args: argparse.Namespace) -> Path:
    from transformers import AutoTokenizer

    try:
        from auto_round import AutoRound
    except ImportError as e:
        print(f"[autoround][ERROR] auto-round import 실패: {e}", file=sys.stderr)
        sys.exit(1)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[autoround] 양자화: {args.hf_model} bits={args.bits} iters={args.iters}")
    autoround = AutoRound(
        model_name=args.hf_model, token=token,
        bits=args.bits, group_size=args.group_size, sym=args.sym,
        nsamples=args.nsamples, seqlen=args.seqlen, iters=args.iters,
        dataset=args.dataset,
    )
    autoround.quantize()
    autoround.save_quantized(str(out_dir), inplace=False, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[autoround] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model, args.bits)
    write_readme(out_dir, model_name, args)
    print(f"[autoround] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: AutoRound INT{args.bits} (group={args.group_size}, iters={args.iters})
---

# {model_name}

`{args.hf_model}` 의 **Intel AutoRound INT{args.bits}** 양자화 (signSGD 반올림 최적화).

## 기술 계보

- **이전: GPTQ/AWQ/HQQ** — 스케일·보상·분해로 양자화 오차를 줄이나, 반올림 자체(0.5 경계)는
  고정된 nearest 규칙에 의존.
- **AutoRound의 개선점**: 각 가중치의 반올림(up/down)을 학습 가능한 변수로 두고
  블록별 재구성 오차를 최소화 → INT4는 물론 INT2에서도 SOTA급 회복.
  HQQ/GPTQ 대비 추가 미세조정 없이 품질 상한 돌파.
- **AutoRound의 단점**: 양자화 시간(iters=200, 8B 수 시간, GPU 필요),
  전용 `auto_round` 포맷 → GPTQ/AWQ 커널과 비호환 구간 존재.
- **AutoRound를 개선하는 후속 기법**: **AutoRound+GPTQ/AWQ 결합**(llmcompressor),
  극저비트는 **AQLM/QuIP#**(벡터·격자 양자화) 병행.

## 파라미터

- bits={args.bits}, group_size={args.group_size}, sym={args.sym},
  nsamples={args.nsamples}, seqlen={args.seqlen}, iters={args.iters}, dataset={args.dataset}

## 사용법

```bash
vllm serve {model_name} --quantization autoround --max-model-len 8192
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
        print("[autoround][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        print("[autoround][ERROR] 업로드 계정 미지정. --hf-user 또는 HF_USER 환경변수로 명시하세요.",
              file=sys.stderr)
        sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add AutoRound quantized model")
        print(f"[autoround] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[autoround][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ AutoRound 추론 가이드 ({model_name}) ================
[1] vLLM 서빙:
    vllm serve {model_name} --quantization autoround --tensor-parallel-size 1 --max-model-len 8192

[2] transformers (+auto-round):
    pip install auto-round accelerate
    from auto_round import AutoRoundConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qconf = AutoRoundConfig(bits={args.bits}, group_size={args.group_size}, sym={str(args.sym)})
    m = AutoModelForCausalLM.from_pretrained("{model_name}", quantization_config=qconf, device_map="auto")
    tok = AutoTokenizer.from_pretrained("{model_name}")
    print(tok.decode(m.generate(tok("Hello, my name is", return_tensors="pt").input_ids.cuda(), max_new_tokens=128)[0]))
================================================================
""")


def main() -> None:
    args = parse_args()
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(args.hf_model, args.bits)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
