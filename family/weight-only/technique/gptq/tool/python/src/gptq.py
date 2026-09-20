# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "auto-gptq",
#   "optimum",
#   "transformers",
#   "torch",
#   "datasets",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///

"""GPTQ 양자화 스크립트 (AutoGPTQ 기반 INT4, Llama-3.1 계열 최적).

사용 예:
    uv run gptq.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-gptq
    uv run gptq.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out --bits 4 --group-size 128 --show-inference-instruction
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoGPTQ INT4 양자화")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True, help="양자화 결과물 저장 디렉터리")
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-GPTQ-Int4-128g")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # 최적 기본값: Llama-3.1 8B + DGX Spark(BF16) 기준 perplexity 최적 조합
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128, help="group-size 128이 Llama 계열 품질/속도 균형 최적")
    p.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True, help="대칭 양자화 (기본 True)")
    p.add_argument("--desc-act", action=argparse.BooleanOptionalAction, default=True, help="desc_act=True가 Llama perplexity 최적")
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--dataset", default="wikitext2", help="캘리브레이션 데이터셋 (wikitext2/c4)")
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def default_output_name(hf_model: str, bits: int, group_size: int) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-GPTQ-Int{bits}-{group_size}g"


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        from autogptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore[no-redef]

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quant_config = BaseQuantizeConfig(
        bits=args.bits, group_size=args.group_size, sym=args.sym,
        desc_act=args.desc_act, damp_percent=args.damp,
    )
    print(f"[gptq] 로드: {args.hf_model} (bf16, DGX Spark 가정)")
    model = AutoGPTQForCausalLM.from_pretrained(
        args.hf_model, quantize_config=quant_config, token=token,
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    # 캘리브레이션 텍스트 준비
    print(f"[gptq] 캘리브레이션: {args.dataset} nsamples={args.nsamples} seqlen={args.seqlen}")
    if args.dataset == "c4":
        raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = [ex["text"] for _, ex in zip(range(args.nsamples), raw)]
    else:
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        joined = "\n\n".join(raw["text"])
        tok = tokenizer(joined, return_tensors="pt")
        ids = tok.input_ids[0]
        texts = [tokenizer.decode(ids[i:i + args.seqlen]) for i in range(0, args.nsamples * args.seqlen, args.seqlen)]
        texts = texts[:args.nsamples]
    examples = [tokenizer(t, return_tensors="pt", truncation=True, max_length=args.seqlen) for t in texts]

    model.quantize(examples, batch_size=args.batch_size, use_triton=False)
    model.save_quantized(str(out_dir), use_safetensors=True)
    tokenizer.save_pretrained(str(out_dir))
    # generation_config / chat template 보존
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json", "config.json", "chat_template*"])
        for name in ("generation_config.json",):
            s = Path(src) / name
            if s.exists() and not (out_dir / name).exists():
                shutil.copy2(s, out_dir / name)
    except Exception as e:
        print(f"[gptq] 부속파일 보존 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model, args.bits, args.group_size)
    write_readme(out_dir, model_name, args)
    print(f"[gptq] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: GPTQ-Int{args.bits} (group={args.group_size}, sym={args.sym}, desc_act={args.desc_act})
---

# {model_name}

`{args.hf_model}` 의 **GPTQ INT{args.bits}** 양자화.
DGX Spark(Blackwell) + bf16 캘리브레이션(seqlen={args.seqlen}, nsamples={args.nsamples}) 기준 최적값.

## 기술 계보

- **이전: 단순 RTN(round-to-nearest) INT8/INT4** — 빠르지만 4bit에서 perplexity 급락,
  특히 Llama 이상치 채널에서 치명적.
- **GPTQ의 개선점**: 레이어별 Hessian 역행렬 기반 오차 보상(OBQ 계열)으로 INT4에서도
  FP16 대비 품질 유지. group_size=128 + desc_act가 핵심.
- **GPTQ의 단점**: 캘리브레이션 데이터 필요, 양자화 시간 수십 분, 활성화 양자화 미지원(W4A16),
  Marlin 커널 없으면 속도 제한.
- **GPTQ를 개선하는 후속 기법**: **AWQ**(이상치 채널 스케일링, instruction 모델 강함),
  **HQQ**(무캘리브 정밀 최적화, 속도), **AutoRound**(반올림 자체를 학습, 2~4bit SOTA),
  **QuaRot/SpinQuant**(회전으로 이상치 제거 후 GPTQ 병행).

## 양자화 파라미터

- bits={args.bits}, group_size={args.group_size}, sym={args.sym}, desc_act={args.desc_act},
  damp={args.damp}, dataset={args.dataset}, seqlen={args.seqlen}

## 사용법

```bash
vllm serve {model_name} --quantization gptq --tensor-parallel-size 1 --max-model-len 8192
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
        print("[gptq][ERROR] --upload 토큰 누락. --hf-token 또는 HF_TOKEN 필요.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        print("[gptq][ERROR] 업로드 계정 미지정. --hf-user 또는 HF_USER 환경변수로 명시하세요.",
              file=sys.stderr)
        sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add GPTQ quantized model")
        print(f"[gptq] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[gptq][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ GPTQ 추론 가이드 ({model_name}) ================
[1] vLLM 서빙:
    vllm serve {model_name} --quantization gptq --tensor-parallel-size 1 --max-model-len 8192

[2] transformers:
    pip install auto-gptq optimum accelerate
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}")
    m = AutoGPTQForCausalLM.from_quantized("{model_name}", device="cuda:0", use_triton=False)
    print(tok.decode(m.generate(tok("Hello, my name is", return_tensors="pt").input_ids.cuda(), max_new_tokens=128)[0]))

[3] OpenAI 호환 호출:
    curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \\
      -d '{{"model":"MODEL","prompt":"Hello","max_tokens":128}}'
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
