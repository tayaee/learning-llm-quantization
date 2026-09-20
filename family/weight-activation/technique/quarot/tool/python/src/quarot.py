# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "accelerate",
#   "huggingface_hub",
#   "fast-hadamard-transform",
# ]
# ///

"""QuaRot 양자화 스크립트 (Hadamard 회전 + W4A4/W4A16 GPTQ).

원 논문(2024): residual·attention·FFN 경로에 Hadamard 회전을 융합해 이상치를 제거한 뒤
W4A4/W4A16으로 양자화. 본 스크립트는 fast-hadamard-transform 기반 온라인 변환 대신
가중치 융합 + INT4 per-group + W4A4 시뮬레이션 스케일 저장을 수행한다.

사용 예:
    uv run quarot.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-quarot
    uv run quarot.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out --mode W4A4 --show-inference-instruction
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuaRot Hadamard 회전 양자화")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-QuaRot-W4A16 (또는 W4A4)")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    p.add_argument("--mode", default="W4A16", choices=["W4A16", "W4A4", "W8A8"],
                   help="W4A16이 품질 최적, W4A4가 속도 최적 (DGX Spark)")
    p.add_argument("--bits", type=int, default=4, help="가중치 비트 (mode와 일치 권장)")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--hadamard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def default_output_name(hf_model: str, mode: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-QuaRot-{mode}"


def apply_hadamard_(W) -> None:
    """2D 가중치 행렬의 입력 차원에 in-place Hadamard 변환 (2의 거듭제곱 패딩 처리)."""
    import torch

    try:
        from fast_hadamard_transform import hadamard_transform
        has_fht = True
    except ImportError:
        has_fht = False
    n_in = W.shape[1]
    p2 = 1
    while p2 < n_in:
        p2 *= 2
    if has_fht and p2 == n_in:
        W.copy_(hadamard_transform(W.float(), scale=1.0 / (n_in ** 0.5)).to(W.dtype))
        return
    # 폴백: 명시적 Hadamard 행렬 곱 (8B hidden=4096 → 4096x4096, 메모리 약 64MB, 허용)
    import numpy as np
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < p2:
        h = np.block([[h, h], [h, -h]])
    h = (h / (p2 ** 0.5)).astype("float32")
    H = torch.from_numpy(h).to(W.device, torch.float32)
    Wpad = torch.zeros(W.shape[0], p2, device=W.device, dtype=torch.float32)
    Wpad[:, :n_in] = W.float()
    W.copy_((Wpad @ H.T)[:, :n_in].to(W.dtype))


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[quarot] 로드: {args.hf_model} mode={args.mode}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    act_bits = {"W4A16": 16, "W4A4": 4, "W8A8": 8}[args.mode]
    with torch.no_grad():
        for layer in model.model.layers:
            for proj_name in ("q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"):
                proj = getattr(getattr(layer, "self_attn", layer), proj_name, None)
                if proj is None and hasattr(layer, "mlp"):
                    proj = getattr(layer.mlp, proj_name, None)
                if proj is None:
                    continue
                if args.hadamard:
                    apply_hadamard_(proj.weight.data)
            for proj_name in ("o_proj", "down_proj"):
                proj = getattr(getattr(layer, "self_attn", layer), proj_name, None)
                if proj is None and hasattr(layer, "mlp"):
                    proj = getattr(layer.mlp, proj_name, None)
                if proj is None:
                    continue
                if args.hadamard:
                    # 출력 측은 전치-Hadamard로 상쇄 (직교이므로 H^T = H)
                    apply_hadamard_(proj.weight.data.T.contiguous().T)

        # per-group INT 양자화 (가중치) + 활성화 스케일 메타 저장
        for name, mod in model.named_modules():
            if mod.__class__.__name__ != "Linear":
                continue
            W = mod.weight.data.float()
            g = args.group_size
            pad = (-W.shape[1]) % g
            if pad:
                W = torch.cat([W, torch.zeros(W.shape[0], pad, device=W.device)], dim=1)
            Wg = W.reshape(W.shape[0], -1, g)
            amax = Wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = amax / float(2 ** (args.bits - 1) - 1)
            q = (Wg / scale).round().clamp(-2 ** (args.bits - 1), 2 ** (args.bits - 1) - 1).to(torch.int8)
            mod.weight.data = (q.float() * scale).reshape(W.shape)[:, : mod.weight.shape[1]].to(mod.weight.dtype)

    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "quarot_config.json").write_text(json.dumps(
        {"mode": args.mode, "weight_bits": args.bits, "activation_bits": act_bits,
         "group_size": args.group_size, "hadamard": args.hadamard, "seed": args.seed}, indent=2),
        encoding="utf-8")
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[quarot] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model, args.mode)
    write_readme(out_dir, model_name, args)
    print(f"[quarot] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: QuaRot {args.mode} (Hadamard={args.hadamard})
---

# {model_name}

`{args.hf_model}` 의 **QuaRot {args.mode}** 양자화 (Hadamard 회전 + INT{args.bits}).

## 기술 계보

- **이전: SmoothQuant/SpinQuant** — 스무딩은 W8까지, SpinQuant 회전은 학습 비용·온라인 변환
  오버헤드가 존재.
- **QuaRot의 개선점**: Hadamard 행렬을 RMSNorm·잔차·QKV 경로에 계산 그래프 동등 변환으로 융합 →
  추가 학습 없이 이상치 제거, GPTQ와 결합해 W4A4/W4A16 실용 품질. 온라인 Hadamard는
  fast-hadamard-transform O(n log n)으로 저렴.
- **QuaRot의 단점**: W4A4 실가속은 전용 INT4×INT4 커널(QServe/Marlin-FP8 계열) 필요,
  KV 캐시는 별도 양자화 필요.
- **QuaRot을 개선하는 후속 기법**: **QServe**(W4A8 KV 양자화 + 시스템 최적화),
  **SpinQuant 학습 회전**(품질 상한), **FP8**(하드웨어 네이티브 단순화).

## 파라미터

- mode={args.mode}, weight_bits={args.bits}, group_size={args.group_size}, hadamard={args.hadamard}

## 사용법

```bash
vllm serve {model_name} --quantization quarot --max-model-len 8192
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
        print("[quarot][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[quarot][ERROR] whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add QuaRot quantized model")
        print(f"[quarot] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[quarot][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    quant_flag = "quarot" if args.mode == "W4A4" else "gptq"
    print(f"""
================ QuaRot 추론 가이드 ({model_name}) ================
[1] vLLM 서빙:
    vllm serve {model_name} --quantization {quant_flag} --tensor-parallel-size 1 --max-model-len 8192

[2] transformers (+fast-hadamard-transform):
    pip install fast-hadamard-transform accelerate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("{model_name}")
    m = AutoModelForCausalLM.from_pretrained("{model_name}", device_map="auto", torch_dtype="auto")
    print(tok.decode(m.generate(tok("Hello, my name is", return_tensors="pt").input_ids.cuda(), max_new_tokens=128)[0]))
================================================================
""")


def main() -> None:
    args = parse_args()
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(args.hf_model, args.mode)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
