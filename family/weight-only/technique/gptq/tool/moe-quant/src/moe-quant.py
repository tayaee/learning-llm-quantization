# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
# ]
# ///

"""GPTQ 양자화 스크립트 — IST-DASLab MoE-Quant 오케스트레이션 (DeepSeek MoE 전용).

공식 파이프라인(https://github.com/IST-DASLab/MoE-Quant):
  1) torchrun quant.py (expert 병렬 Hessian + triton 고속 GPTQ, 4bit 대칭)
  2) pack_quantized_model.py → compressed_tensors (transformers/vLLM 호환)
R1-0528 실측 recovery 99.82%, AWQ(94.29%)를 상회한다.

주의: MoE-Quant는 DeepSeek-V3/R1 계열에 검증됐다. V4 신아키텍처
(CSA+HCA 하이브리드 어텐션, mHC)에 대한 호환성은 미확인이며,
실패 시 저장소 업데이트 또는 llmcompressor MoE 레시피로 전환하라.

사용 예:
    uv run moe-quant.py --hf-model deepseek-ai/DeepSeek-V4-Flash-Base --output-dir ./out-v4-gptq
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/IST-DASLab/MoE-Quant.git"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSeek MoE 전용 GPTQ (MoE-Quant)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: deepseek-ai/DeepSeek-V4-Flash-Base)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-GPTQ-4b-128g-experts")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # 공식 README 검증값 (R1-0528 최고 회복 조합)
    p.add_argument("--dataset", default="open-thoughts",
                   choices=["open-thoughts", "open-platypus", "fineweb-edu"])
    p.add_argument("--nsamples", type=int, default=512)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--bits", type=int, default=4, help="MoE-Quant는 4bit 대칭만 지원")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--quantization-order", default="activation", choices=["default", "activation"])
    p.add_argument("--quantization-scale", default="mse", choices=["absmax", "mse"])
    p.add_argument("--quantize-only-experts", action=argparse.BooleanOptionalAction, default=True,
                   help="비공유 expert만 양자화 (최고 recovery, 권장)")
    p.add_argument("--num-gpus", type=int, default=8, help="expert 병렬 프로세스 수 (8xH100 권장)")
    p.add_argument("--repo-dir", default="third_party/MoE-Quant")
    return p.parse_args()


def default_output_name(hf_model: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-GPTQ-4b-128g-experts"


def ensure_repo(repo_dir: Path) -> Path:
    if not (repo_dir / "quant.py").exists():
        print(f"[moe-quant] 체크아웃: {REPO_URL} → {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    else:
        print(f"[moe-quant] 기존 체크아웃 사용: {repo_dir}")
    return repo_dir


def run_quantization(args: argparse.Namespace) -> Path:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quant_dir = out_dir / "quantized"
    quant_dir.mkdir(parents=True, exist_ok=True)

    # 공식 요구: --model_name_or_path는 로컬 스냅샷 "정확 경로"
    if os.path.isdir(args.hf_model):
        snap = args.hf_model
    else:
        print(f"[moe-quant] 스냅샷 다운로드: {args.hf_model} (284GB급, 시간 소요)")
        snap = snapshot_download(repo_id=args.hf_model, token=token)
    print(f"[moe-quant] 입력 스냅샷: {snap}")

    repo_dir = Path(args.repo_dir)
    if not repo_dir.is_absolute():
        repo_dir = (Path.cwd() / repo_dir).resolve()
    ensure_repo(repo_dir)

    cmd = ["torchrun", f"--nnodes=1", f"--nproc-per-node={args.num_gpus}",
           "--master_port", os.environ.get("MASTER_PORT", "29501"), "quant.py",
           "--model_name_or_path", snap,
           "--dataset_name_or_path", args.dataset,
           "--num_calibration_samples", str(args.nsamples),
           "--max_sequence_length", str(args.seqlen),
           "--bits", str(args.bits),
           "--group_size", str(args.group_size),
           "--rel_damp", "0.1",
           "--sym",
           "--offload_activations",
           "--quantization_order", args.quantization_order,
           "--quantization_scale", args.quantization_scale,
           "--tie_gptq_handles",
           "--dtype", "bfloat16",
           "--save_dir", str(quant_dir)]
    if args.quantize_only_experts:
        cmd.append("--quantize_only_experts")
    print(f"[moe-quant] 양자화 실행 ({args.num_gpus} GPU, 수 시간 소요)")
    subprocess.run(cmd, cwd=str(repo_dir), check=True)

    pack_cmd = [sys.executable, "pack_quantized_model.py",
                "--model_name_or_path", snap,
                "--quantized_model_path", str(quant_dir),
                "--packed_model_path", str(out_dir / "packed"),
                "--dtype", "bfloat16"]
    print("[moe-quant] compressed_tensors 패킹")
    subprocess.run(pack_cmd, cwd=str(repo_dir), check=True)

    # 패킹 결과를 output 루트로 승격 + 토크나이저 보존
    packed = out_dir / "packed"
    for item in packed.iterdir():
        dest = out_dir / item.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    packed.rmdir()
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    tok.save_pretrained(str(out_dir))
    for name in ("generation_config.json", "chat_template.jinja"):
        s = Path(snap) / name
        if s.exists() and not (out_dir / name).exists():
            shutil.copy2(s, out_dir / name)

    model_name = args.output_model_name or default_output_name(args.hf_model)
    write_readme(out_dir, model_name, args)
    print(f"[moe-quant] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: GPTQ INT{args.bits} symmetric via MoE-Quant (compressed-tensors)
---

# {model_name}

`{args.hf_model}` 의 **GPTQ INT{args.bits}** 양자화 (MoE-Quant, expert 병렬).

## 기술 계보

- **이전: AWQ 일괄 적용** — R1 실측 recovery 94.29%에 그치고 MoE expert별
  캘리브레이션 편향에 취약 (Flash AWQ 커뮤니티 시도도 ~17점 MMLU 손실).
- **MoE-Quant의 개선점**: expert 샤딩 Hessian + triton 고속 GPTQ(10×) +
  데이터 병렬로 671B를 8×H100 2시간에 처리, experts-only 변형이
  R1-0528에서 99.82% recovery. 출력은 vLLM 직행 compressed_tensors.
- **단점**: 4bit 대칭만 지원, V3/R1 검증 (V4 신아키텍처 호환 미확인),
  8 GPU 없이는 실행 불가.
- **대안**: 공식 NVFP4(`nvidia/DeepSeek-V4-Flash-NVFP4`, Blackwell 전용),
  단일 GPU면 GGUF IQ 계열.

## 파라미터

- dataset={args.dataset}, nsamples={args.nsamples}, seqlen={args.seqlen},
  group_size={args.group_size}, order={args.quantization_order},
  scale={args.quantization_scale}, experts_only={args.quantize_only_experts}

## 사용법 (8×H100 단일 노드)

```bash
vllm serve {model_name} --tensor-parallel-size 8 --trust-remote-code --max-model-len 32768
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
        print("[moe-quant][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[moe-quant][ERROR] whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id,
                          ignore_patterns=["quantized/**"],
                          commit_message="Add GPTQ quantized model (MoE-Quant)")
        print(f"[moe-quant] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[moe-quant][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ MoE-Quant GPTQ 추론 가이드 ({model_name}) ================
[1] vLLM 서빙 (8×H100 단일 노드, compressed-tensors):
    vllm serve {model_name} --tensor-parallel-size 8 --trust-remote-code --max-model-len 32768

[2] transformers:
    pip install compressed-tensors accelerate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}", trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained("{model_name}", device_map="auto",
                                             torch_dtype="auto", trust_remote_code=True)
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
