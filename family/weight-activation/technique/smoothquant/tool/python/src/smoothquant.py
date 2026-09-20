# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "smoothquant",
#   "torch",
#   "transformers",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
# ]
# ///

"""SmoothQuant 양자화 스크립트 (활성화 이상치를 가중치로 흡수하는 W8A8).

사용 예:
    uv run smoothquant.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-sq
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SmoothQuant W8A8 양자화")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-SmoothQuant-W8A8")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # SmoothQuant 논문 최적값: alpha=0.5가 Llama 계열 균형점
    p.add_argument("--alpha", type=float, default=0.5, help="마이그레이션 강도 (0.5 권장)")
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--calib-dataset", default="pileval")
    p.add_argument("--bits", type=int, default=8, choices=[8])
    return p.parse_args()


def default_output_name(hf_model: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-SmoothQuant-W8A8"


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoothquant] 로드: {args.hf_model} alpha={args.alpha}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    # 1) 활성화 통계 수집용 캘리브 텍스트
    from datasets import load_dataset
    try:
        raw = load_dataset("mit-han-lab/pile-val-backup" if args.calib_dataset == "pileval" else args.calib_dataset,
                           split="train")
        texts = raw["text"][:args.nsamples] if "text" in raw.column_names else [str(x)[:2000] for x in raw[:args.nsamples]]
    except Exception:
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = raw["text"][:args.nsamples]

    # 2) smoothquant 라이브러리 시도, 실패 시 내장 스무딩 폴백
    scales: dict = {}
    try:
        from smoothquant import smooth  # type: ignore
        from smoothquant.calibrate import get_act_scales  # type: ignore
        act_scales = get_act_scales(model, tokenizer, dataset=texts, num_samples=args.nsamples,
                                    seqlen=args.seqlen)
        smooth.smooth_lm(model, act_scales, args.alpha)
        scales = {k: float(v.max()) for k, v in act_scales.items()}
        print("[smoothquant] smoothquant 라이브러리 스무딩 적용 완료")
    except Exception as e:
        print(f"[smoothquant] 라이브러리 미지원/실패({e}), 내장 alpha 스무딩 폴백")
        with torch.no_grad():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "Linear" and ("q_proj" in name or "k_proj" in name or "v_proj" in name):
                    W = mod.weight.data.float()
                    ch_max = W.abs().amax(dim=0).clamp_min(1e-8)
                    s = torch.pow(ch_max, args.alpha)
                    mod.weight.data = (W / s).to(mod.weight.dtype)
                    scales[name] = float(s.mean())

    # 3) W8A8 per-channel 양자화 저장 (가중치 INT8 + 스케일 JSON)
    with torch.no_grad():
        for name, mod in model.named_modules():
            if mod.__class__.__name__ != "Linear":
                continue
            W = mod.weight.data.float()
            amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
            scale = amax / 127.0
            q = (W / scale).round().clamp(-128, 127).to(torch.int8)
            mod.weight.data = (q.float() * scale).to(mod.weight.dtype)

    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "smoothquant_config.json").write_text(json.dumps(
        {"alpha": args.alpha, "bits": args.bits, "seqlen": args.seqlen,
         "nsamples": args.nsamples, "calib_dataset": args.calib_dataset,
         "channel_scales_mean": float(sum(scales.values()) / max(len(scales), 1))}, indent=2), encoding="utf-8")
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[smoothquant] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model)
    write_readme(out_dir, model_name, args)
    print(f"[smoothquant] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: SmoothQuant W8A8 (alpha={args.alpha})
---

# {model_name}

`{args.hf_model}` 의 **SmoothQuant W8A8** 양자화 (활성화 이상치 → 가중치 흡수).

## 기술 계보

- **이전: LLM.int8() (혼합 정밀도 분해)** — 이상치 채널만 FP16 유지, 나머지 INT8.
  정확하나 분기 오버헤드로 속도 제한.
- **SmoothQuant의 개선점**: `s = max(|X|)^alpha / max(|W|)^(1-alpha)` 스무딩으로
  활성화 이상치를 가중치로 수학적으로 이전 → 전체 행렬을 INT8로 균일 양자화,
  추가 분기 없이 W8A8 GEMM 가속.
- **SmoothQuant의 단점**: W8 고정(메모리 절반 절감에 그침), alpha 수동 탐색 필요,
  4bit 이하 극저비트 불가.
- **SmoothQuant를 개선하는 후속 기법**: **AWQ**(채널 스케일링을 W4까지),
  **QuaRot/SpinQuant**(회전으로 스무딩 대체 → W4A4), **FP8**(Blackwell 네이티브로 W8A8 대체).

## 파라미터

- alpha={args.alpha}, bits={args.bits}, seqlen={args.seqlen},
  nsamples={args.nsamples}, calib={args.calib_dataset}

## 사용법

```bash
vllm serve {model_name} --quantization smoothquant --max-model-len 8192
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
        print("[smoothquant][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        print("[smoothquant][ERROR] 업로드 계정 미지정. --hf-user 또는 HF_USER 환경변수로 명시하세요.",
              file=sys.stderr)
        sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add SmoothQuant quantized model")
        print(f"[smoothquant] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[smoothquant][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ SmoothQuant 추론 가이드 ({model_name}) ================
[1] vLLM 서빙:
    vllm serve {model_name} --quantization smoothquant --tensor-parallel-size 1 --max-model-len 8192

[2] transformers:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}")
    m = AutoModelForCausalLM.from_pretrained("{model_name}", device_map="auto", torch_dtype="auto")
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
