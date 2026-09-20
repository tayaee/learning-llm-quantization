# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "autoawq",
#   "transformers",
#   "torch",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///

"""AWQ 양자화 스크립트 (Activation-aware W4A16, Marlin 커널 호환).

사용 예:
    uv run awq.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-awq
    uv run awq.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out --w-bit 4 --q-group-size 128 --show-inference-instruction
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AWQ W4A16 양자화 (Marlin 호환)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-AWQ-Int4-128g")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # AWQ 최적 기본값 (Llama-3.1 8B, DGX Spark)
    p.add_argument("--w-bit", type=int, default=4, choices=[4])
    p.add_argument("--q-group-size", type=int, default=128)
    p.add_argument("--zero-point", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--version", default="GEMM", choices=["GEMM", "GEMV"], help="GEMM=Marlin 호환 패킹 (처리량 최적)")
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--calib-data", default="pileval", help="pileval이 AWQ 논문 표준")
    return p.parse_args()


def default_output_name(hf_model: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-AWQ-Int4-128g"


def run_quantization(args: argparse.Namespace) -> Path:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quant_config = {"zero_point": args.zero_point, "q_group_size": args.q_group_size,
                    "w_bit": args.w_bit, "version": args.version}
    print(f"[awq] 로드: {args.hf_model} config={quant_config}")
    model = AutoAWQForCausalLM.from_pretrained(args.hf_model, token=token, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    print(f"[awq] 양자화 실행 (calib={args.calib_data}, seqlen={args.seqlen})")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=args.calib_data,
                   n_samples=args.nsamples, seqlen=args.seqlen)
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json", "config.json"])
        for name in ("generation_config.json",):
            s = Path(src) / name
            if s.exists() and not (out_dir / name).exists():
                shutil.copy2(s, out_dir / name)
    except Exception as e:
        print(f"[awq] 부속파일 보존 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model)
    write_readme(out_dir, model_name, args)
    print(f"[awq] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: AWQ W4A16 (w_bit={args.w_bit}, group={args.q_group_size}, version={args.version})
---

# {model_name}

`{args.hf_model}` 의 **AWQ INT4 (W4A16)** 양자화. Marlin/GEMM 패킹으로 DGX Spark GPU 처리량 최적.

## 기술 계보

- **이전: GPTQ** — Hessian 보상으로 INT4 품질을 확보했으나 instruction-following 열화가
  Llama 계열에서 관찰됨, 캘리브 누적오차 존재.
- **AWQ의 개선점**: 활성화 분포를 보고 중요 가중치 채널을 스케일 업(activation-aware scaling)하여
  양자화 전 saliency 보존. 소량 캘리브(pileval 128)로 GPTQ 대비 instruction 모델 강함 + Marlin 융합.
- **AWQ의 단점**: W4A16이라 활성화는 FP16 그대로(메모리 대역폭 절반 절감에 그침),
  W4A4/W8A8 같은 저비트 활성화 가속 불가.
- **AWQ를 개선하는 후속 기법**: **QuaRot/SpinQuant**(회전으로 이상치 제거 → W4A4까지),
  **SmoothQuant**(활성화를 가중치로 이동 → W8A8), **FP8-E4M3**(Blackwell 네이티브, vLLM 기본),
  **HQQ/AutoRound**(무캘리브·반올림학습으로 품질 상한 돌파).

## 양자화 파라미터

- w_bit={args.w_bit}, q_group_size={args.q_group_size}, zero_point={args.zero_point},
  version={args.version}, calib={args.calib_data}, seqlen={args.seqlen}

## 사용법

```bash
vllm serve {model_name} --quantization awq --tensor-parallel-size 1 --max-model-len 8192
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
        print("[awq][ERROR] --upload 토큰 누락. --hf-token 또는 HF_TOKEN 필요.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[awq][ERROR] --hf-user 생략 + whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add AWQ quantized model")
        print(f"[awq] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[awq][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ AWQ 추론 가이드 ({model_name}) ================
[1] vLLM 서빙 (Marlin):
    vllm serve {model_name} --quantization awq --tensor-parallel-size 1 --max-model-len 8192

[2] transformers:
    pip install autoawq accelerate
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}")
    m = AutoAWQForCausalLM.from_quantized("{model_name}", fuse_layers=True).cuda()
    print(tok.decode(m.generate(tok("Hello, my name is", return_tensors="pt").input_ids.cuda(), max_new_tokens=128)[0]))
================================================================
""")


def main() -> None:
    args = parse_args()
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(args.hf_model)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
