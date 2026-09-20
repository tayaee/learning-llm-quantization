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

"""FP8 양자화 스크립트 (네이티브 E4M3, Blackwell Tensor Core + vLLM/SGLang 호환).

사용 예:
    uv run fp8.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-fp8
    uv run fp8.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out --fp8-format e4m3n --show-inference-instruction
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="네이티브 FP8 (E4M3/E5M2) 양자화")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-FP8-E4M3")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # DGX Spark Blackwell 기준: E4M3(가중치/활성화) + dynamic이 품질/속도 최적
    p.add_argument("--fp8-format", default="e4m3n", choices=["e4m3n", "e5m2"],
                   help="e4m3n이 LLM 가중치 기본 (정밀도 우선)")
    p.add_argument("--activation-scheme", default="dynamic", choices=["dynamic", "static"],
                   help="dynamic이 캘리브 불필요·분포 강건")
    p.add_argument("--weight-block-size", type=int, default=128, help="per-block FP8 스케일 (128 권장)")
    p.add_argument("--calib-dataset", default="ultrachat", help="static 스킴 선택 시에만 사용")
    p.add_argument("--calib-nsamples", type=int, default=512)
    return p.parse_args()


def default_output_name(hf_model: str, fp8_format: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    tag = "E4M3" if fp8_format == "e4m3n" else "E5M2"
    return f"{base}-FP8-{tag}"


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as e:
        print(f"[fp8][ERROR] llmcompressor import 실패: {e}", file=sys.stderr)
        sys.exit(1)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fp8] 로드: {args.hf_model} (Blackwell FP8, scheme={args.activation_scheme})")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    modifier = QuantizationModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "input_activations": {"num_bits": 8, "type": "float",
                                      "strategy": "tensor" if args.activation_scheme == "static" else "token",
                                      "dynamic": args.activation_scheme == "dynamic"},
                "weights": {"num_bits": 8, "type": "float", "strategy": "block",
                            "block_structure": [args.weight_block_size, 1],
                            "dynamic": False},
            }
        },
        ignore=["lm_head"],
    )
    calib_data = None
    if args.activation_scheme == "static":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train").select(range(args.calib_nsamples))
        calib_data = ds.map(lambda x: {"text": tokenizer.apply_chat_template(
            x["messages"], tokenize=False, add_generation_prompt=False)})
    oneshot(model=model, dataset=calib_data, recipe=modifier,
            max_seq_length=2048, num_calibration_samples=args.calib_nsamples if calib_data else 0)

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
        print(f"[fp8] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model, args.fp8_format)
    write_readme(out_dir, model_name, args)
    print(f"[fp8] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: FP8-{args.fp8_format} (activation={args.activation_scheme})
---

# {model_name}

`{args.hf_model}` 의 **네이티브 FP8({args.fp8_format})** 양자화. Blackwell Tensor Core 가속.

## 기술 계보

- **이전: W8A8 INT (SmoothQuant/LLM.int8)** — INT8 커널 의존, 이상치 처리용 스무딩 필요,
  Blackwell에서는 FP8 대비 효율 저하.
- **FP8의 개선점**: E4M3(정밀도)/E5M2(범위) 부동소수 그대로 연산 → 스무딩·이상치 흡수 불필요,
  dynamic per-token 스케일로 분포 강건, vLLM/SGLang 네이티브 지원으로 DGX Spark에서 즉시 가속.
- **FP8의 단점**: 메모리 절감 2배에 그침(INT4 대비 큼), 구형 GPU(Ampere 이하) 미지원,
  7B 이하 소형 모델에서 INT4 대비 품질 이득이 작을 수 있음.
- **FP8을 개선하는 후속 기법**: 극저비트 필요 시 **GPTQ/AWQ INT4**(메모리),
  품질 한계 돌파는 **FP8 + INT4 혼합(attention FP8 + MLP INT4)**.

## 파라미터

- fp8_format={args.fp8_format}, activation_scheme={args.activation_scheme},
  weight_block_size={args.weight_block_size}

## 사용법

```bash
vllm serve {model_name} --quantization fp8 --tensor-parallel-size 1 --max-model-len 8192
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
        print("[fp8][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[fp8][ERROR] whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add FP8 quantized model")
        print(f"[fp8] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[fp8][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ FP8 추론 가이드 ({model_name}) ================
[1] vLLM 서빙 (Blackwell 권장):
    vllm serve {model_name} --quantization fp8 --tensor-parallel-size 1 --max-model-len 8192

[2] SGLang:
    python -m sglang.launch_server --model-path {model_name} --quantization fp8 --port 30000

[3] transformers:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}")
    m = AutoModelForCausalLM.from_pretrained("{model_name}", device_map="auto", torch_dtype="auto")
    print(tok.decode(m.generate(tok("Hello, my name is", return_tensors="pt").input_ids.cuda(), max_new_tokens=128)[0]))
================================================================
""")


def main() -> None:
    args = parse_args()
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(args.hf_model, args.fp8_format)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
