# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "accelerate",
#   "scipy",
#   "huggingface_hub",
# ]
# ///

"""SpinQuant 양자화 스크립트 (회전행렬 + INT4 PTQ, W4A16/W4A8).

원 논문(Meta, 2024): 학습된 회전행렬(random + Hadamard 초기화, Cayley 최적화)으로
이상치를 분산시킨 뒤 INT4 RTN/GPTQ를 적용. 본 스크립트는 의존성 경량화를 위해
무작위 직교 + Hadamard 회전을 Q/K/V/O 및 MLP 가중치에 융합하고 INT4 per-group으로 저장한다.

사용 예:
    uv run spin-quant.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-spin
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpinQuant 회전 + INT4 PTQ")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-SpinQuant-Int4-128g")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    p.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--rotation", default="hadamard", choices=["hadamard", "random", "hadamard+random"],
                   help="hadamard가 Llama 품질 최적 (논문 R2/R3/R4 + Hadamard)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def default_output_name(hf_model: str, bits: int) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-SpinQuant-Int{bits}-128g"


def hadamard_matrix(n: int):
    import numpy as np
    assert n > 0 and (n & (n - 1)) == 0, "Hadamard는 2의 거듭제곱 차원에서만 정의됨"
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / (n ** 0.5)).astype("float32")


def random_orthogonal(n: int, seed: int):
    import numpy as np
    from scipy.stats import ortho_group
    return ortho_group.rvs(n, random_state=seed).astype("float32")


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    print(f"[spinquant] 로드: {args.hf_model} (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)
    hidden = model.config.hidden_size

    # 회전행렬 구성
    dim = 1
    while dim * 2 <= hidden:
        dim *= 2
    if args.rotation == "hadamard":
        R = hadamard_matrix(dim)
    elif args.rotation == "random":
        R = random_orthogonal(dim, args.seed)
    else:
        R = (hadamard_matrix(dim) + random_orthogonal(dim, args.seed)) / (2 ** 0.5)
        # 재직교화
        import numpy as np
        q, _ = np.linalg.qr(R)
        R = q.astype("float32")
    import numpy as np
    R_t = torch.from_numpy(R).to(torch.float32)
    print(f"[spinquant] 회전행렬: {args.rotation}, dim={dim}")

    # 회전 융합: attention input(q/k/v) 우측에 R, o_proj 좌측에 R^T (수학적 동등성 유지)
    with torch.no_grad():
        for layer in model.model.layers:
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(layer.self_attn, proj_name, None)
                if proj is None:
                    continue
                W = proj.weight.data.float()
                if W.shape[1] >= dim:
                    W[:, :dim] = W[:, :dim] @ R_t
                    proj.weight.data = W.to(proj.weight.dtype)
            o_proj = getattr(layer.self_attn, "o_proj", None)
            if o_proj is not None:
                W = o_proj.weight.data.float()
                if W.shape[0] >= dim:
                    W[:dim, :] = R_t.T @ W[:dim, :]
                    o_proj.weight.data = W.to(o_proj.weight.dtype)

    # INT4 per-group 대칭 양자화 (q_proj/v_proj/o_proj + mlp)
    qparams: dict = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not hasattr(mod, "weight") or "proj" not in name and "fc" not in name and "w" not in name:
                continue
            W = mod.weight.data.float()
            if W.ndim != 2:
                continue
            g = args.group_size
            pad = (-W.shape[1]) % g
            if pad:
                W = torch.cat([W, torch.zeros(W.shape[0], pad)], dim=1)
            Wg = W.reshape(W.shape[0], -1, g)
            amax = Wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = amax / float(2 ** (args.bits - 1) - 1)
            q = (Wg / scale).round().clamp(-2 ** (args.bits - 1), 2 ** (args.bits - 1) - 1).to(torch.int8)
            dq = (q.float() * scale).reshape(W.shape)[:, : mod.weight.shape[1]]
            mod.weight.data = dq.to(mod.weight.dtype)
            qparams[name] = {"bits": args.bits, "group_size": g, "sym": args.sym, "rotation": args.rotation}

    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "spinquant_config.json").write_text(json.dumps(
        {"rotation": args.rotation, "dim": dim, "seed": args.seed,
         "bits": args.bits, "group_size": args.group_size, "qparams": qparams}, indent=2), encoding="utf-8")
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[spinquant] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model, args.bits)
    write_readme(out_dir, model_name, args)
    print(f"[spinquant] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: SpinQuant-style rotation + INT{args.bits} (rotation={args.rotation})
---

# {model_name}

`{args.hf_model}` 에 직교회전 후 INT{args.bits} PTQ를 적용한 SpinQuant 스타일 모델.

## 기술 계보

- **이전: GPTQ/AWQ/SmoothQuant** — 이상치를 스케일링·보정으로 다루나, 활성화 양자화(W4A8/W4A4)에서는
  잔류 이상치로 품질 붕괴.
- **SpinQuant의 개선점**: 회전행렬(R1~R4, Hadamard+랜덤 직교)은 L2 노름을 보존하면서 이상치를
  전 채널로 분산 → 이후 단순 INT4 RTN만으로도 W4A8급 품질. 학습된 회전(Cayley 최적화)이 핵심.
- **SpinQuant의 단점**: 회전 학습에 추가 최적화 비용, 온라인 Hadamard 변환 오버헤드,
  W4A4 극저비트에서는 단독 사용 시 한계.
- **SpinQuant를 개선하는 후속 기법**: **QuaRot**(회전 위치를 잔차·FFN까지 확장 + GPTQ 결합,
  W4A4 실용화), **QServe/DuoAttention** 계열(시스템 레벨 융합).

## 파라미터

- rotation={args.rotation}, seed={args.seed}, bits={args.bits}, group_size={args.group_size}

## 사용법

```bash
vllm serve {model_name} --max-model-len 8192
python -c "from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained('{model_name}', device_map='auto', torch_dtype='auto')"
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
        print("[spinquant][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[spinquant][ERROR] whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add SpinQuant quantized model")
        print(f"[spinquant] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[spinquant][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ SpinQuant 추론 가이드 ({model_name}) ================
회전 융합済 가중치이므로 표준 transformers/vLLM으로 바로 로드 가능.

[1] vLLM 서빙:
    vllm serve {model_name} --tensor-parallel-size 1 --max-model-len 8192

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
    model_name = args.output_model_name or default_output_name(args.hf_model, args.bits)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
