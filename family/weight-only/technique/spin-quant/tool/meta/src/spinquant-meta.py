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

"""SpinQuant 양자화 스크립트 — Meta 공식 구현 오케스트레이션 (학습된 회전).

공식 파이프라인(facebookresearch/SpinQuant):
  1) bash scripts/10_optimize_rotation.sh $model $w $a $kv   (Cayley SGD 회전 학습)
  2) 학습된 회전(R1/R2)을 HF 가중치에 융합 → INT4 per-group → 저장
논문: "SpinQuant_no_had는 회전된 가중치로 교체하는 것만으로 충분"하므로,
융합된 가중치는 추가 커널 없이 표준 transformers/vLLM으로 로드된다.

사용 예:
    uv run spinquant-meta.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-sq-meta
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/facebookresearch/SpinQuant.git"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpinQuant 공식 학습 파이프라인 (Meta)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-SpinQuant-W4A8-meta")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # 공식 스크립트 인자 (scripts/10_optimize_rotation.sh $model $w $a $kv)
    p.add_argument("--w-bits", type=int, default=4)
    p.add_argument("--a-bits", type=int, default=8, help="W4A8이 no_had 최적 (논문 Table 1)")
    p.add_argument("--kv-bits", type=int, default=8)
    p.add_argument("--fsdp", action="store_true", help="70B급: 11_optimize_rotation_fsdp.sh 사용")
    p.add_argument("--repo-dir", default="third_party/SpinQuant", help="공식 저장소 체크아웃 경로")
    p.add_argument("--install-deps", action=argparse.BooleanOptionalAction, default=True,
                   help="공식 requirement.txt 설치 (기본 True)")
    p.add_argument("--rotation-path", default=None, help="학습된 회전 체크포인트 직접 지정 (생략 시 자동 탐색)")
    p.add_argument("--bits", type=int, default=4, help="융합 후 가중치 INT 비트")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--no-quantize", action="store_true", help="융합만 하고 INT 양자화 생략 (순수 회전 fp16)")
    return p.parse_args()


def default_output_name(hf_model: str, w_bits: int, a_bits: int) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-SpinQuant-W{w_bits}A{a_bits}-meta"


def ensure_repo(repo_dir: Path, install_deps: bool) -> Path:
    if not (repo_dir / "optimize_rotation.py").exists():
        print(f"[spinquant-meta] 체크아웃: {REPO_URL} → {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    else:
        print(f"[spinquant-meta] 기존 체크아웃 사용: {repo_dir}")
    if install_deps and (repo_dir / "requirement.txt").exists():
        print("[spinquant-meta] 공식 requirement.txt 설치")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(repo_dir / "requirement.txt")], check=True)
    return repo_dir


def run_official_optimize(args: argparse.Namespace, repo_dir: Path) -> None:
    script = "11_optimize_rotation_fsdp.sh" if args.fsdp else "10_optimize_rotation.sh"
    cmd = ["bash", f"scripts/{script}", args.hf_model,
           str(args.w_bits), str(args.a_bits), str(args.kv_bits)]
    print(f"[spinquant-meta] 회전 학습 실행: {' '.join(cmd)} (작업 디렉터리: {repo_dir})")
    subprocess.run(cmd, cwd=str(repo_dir), check=True)


def find_rotation_checkpoint(repo_dir: Path) -> Path:
    cands = sorted(repo_dir.rglob("*.pt")) + sorted(repo_dir.rglob("*.pth"))
    cands = [c for c in cands if "rotat" in c.name.lower()]
    if not cands:
        raise FileNotFoundError(
            f"{repo_dir}에서 회전 체크포인트(*rotat*.pt)를 찾지 못했습니다. "
            "--rotation-path로 직접 지정하세요.")
    ckpt = max(cands, key=lambda p: p.stat().st_mtime)
    print(f"[spinquant-meta] 회전 체크포인트: {ckpt}")
    return ckpt


def fuse_rotations(model, ckpt_path: Path) -> dict:
    """학습된 R1/R2를 HF 가중치에 융합. 적용 내역 반환, 매칭 실패 시 명시적 종료."""
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        print(f"[spinquant-meta][ERROR] 체크포인트 형식 미지원: {type(ckpt)}", file=sys.stderr)
        sys.exit(1)

    per_layer: dict[tuple[int, str], torch.Tensor] = {}
    global_rots: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if not isinstance(val, torch.Tensor) or val.ndim != 2 or val.shape[0] != val.shape[1]:
            continue
        n = key.lower().replace("_", "")
        rot = ("R1" if re.search(r"(^|\D)r1(\D|$)", n) else
               "R2" if re.search(r"(^|\D)r2(\D|$)", n) else None)
        if rot is None:
            continue
        m = re.search(r"layers?\D*(\d+)", n) or re.search(r"\.(\d+)\.", key)
        if m:
            per_layer[(int(m.group(1)), rot)] = val.float()
        elif rot not in global_rots:
            global_rots[rot] = val.float()

    layers = model.model.layers
    applied: dict[str, int] = {"R1": 0, "R2": 0}
    max_orth_err = 0.0
    with torch.no_grad():
        for i, layer in enumerate(layers):
            for rot, (in_projs, out_projs) in (
                ("R1", (("q_proj", "k_proj", "v_proj"), ("o_proj",))),
                ("R2", (("up_proj", "gate_proj"), ("down_proj",))),
            ):
                R = per_layer.get((i, rot), global_rots.get(rot))
                if R is None:
                    continue
                orth_err = float((R @ R.T - torch.eye(R.shape[0])).abs().max())
                max_orth_err = max(max_orth_err, orth_err)
                if orth_err > 1e-2:
                    print(f"[spinquant-meta] 경고: layer {i} {rot} 직교성 오차 {orth_err:.2e}")
                attn, mlp = layer.self_attn, layer.mlp
                for pname in in_projs:
                    mod = getattr(attn, pname, getattr(mlp, pname, None))
                    if mod is not None and mod.weight.shape[1] == R.shape[0]:
                        mod.weight.data = (mod.weight.float() @ R.to(mod.weight.device)).to(mod.weight.dtype)
                for pname in out_projs:
                    mod = getattr(attn, pname, getattr(mlp, pname, None))
                    if mod is not None and mod.weight.shape[0] == R.shape[0]:
                        mod.weight.data = (R.T.to(mod.weight.device) @ mod.weight.float()).to(mod.weight.dtype)
                applied[rot] += 1
    if applied["R1"] == 0 and applied["R2"] == 0:
        print("[spinquant-meta][ERROR] 체크포인트에서 R1/R2를 매칭하지 못했습니다. 키 목록:",
              file=sys.stderr)
        for k in list(state.keys())[:50]:
            print(f"  - {k}", file=sys.stderr)
        sys.exit(1)
    print(f"[spinquant-meta] 융합 완료: R1 {applied['R1']}층, R2 {applied['R2']}층, "
          f"최대 직교성 오차 {max_orth_err:.2e}")
    return {"R1_layers": applied["R1"], "R2_layers": applied["R2"],
            "max_orthogonality_error": max_orth_err}


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(args.repo_dir)
    if not repo_dir.is_absolute():
        repo_dir = (Path.cwd() / repo_dir).resolve()
    if args.rotation_path is None:
        ensure_repo(repo_dir, args.install_deps)
        run_official_optimize(args, repo_dir)
        ckpt_path = find_rotation_checkpoint(repo_dir)
    else:
        ckpt_path = Path(args.rotation_path)

    print(f"[spinquant-meta] HF 모델 로드 및 회전 융합: {args.hf_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)
    fusion_info = fuse_rotations(model, ckpt_path)

    if not args.no_quantize:
        with torch.no_grad():
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
    (out_dir / "spinquant_meta_config.json").write_text(json.dumps(
        {"w_bits": args.w_bits, "a_bits": args.a_bits, "kv_bits": args.kv_bits,
         "rotation_checkpoint": str(ckpt_path), "fusion": fusion_info,
         "quantized_bits": None if args.no_quantize else args.bits,
         "group_size": args.group_size}, indent=2), encoding="utf-8")
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[spinquant-meta] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model, args.w_bits, args.a_bits)
    write_readme(out_dir, model_name, args)
    print(f"[spinquant-meta] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: SpinQuant official (learned R1/R2, W{args.w_bits}A{args.a_bits}) + INT{args.bits}
---

# {model_name}

`{args.hf_model}` 의 **SpinQuant 공식 학습 파이프라인** 결과물
(Cayley SGD 학습 회전 융합 + INT{args.bits}).

## 기술 계보 (도구 관점)

- **이전: 고정 회전(QuaRot식 Hadamard)** — 학습 없이 빠르지만 회전 선택에 따라
  품질 편차가 크다(최대 13점 차이).
- **본 도구의 개선점**: Meta 공식 `optimize_rotation.py`로 W{args.w_bits}A{args.a_bits} 목적에
  맞는 회전을 직접 학습 → 고정 회전 대비 안정적 SOTA. R1/R2 융합(no_had)은
  포워드 변경·전용 커널 없이 표준 로드 가능.
- **단점**: 회전 학습에 8B 수십 분(GPU), 70B 수 시간 + FSDP 필요.
- **대안**: 무캘리브 고속은 `tool/llmcompressor/`, 경량 자작은 `tool/python/`.

## 파라미터

- w_bits={args.w_bits}, a_bits={args.a_bits}, kv_bits={args.kv_bits}, fsdp={args.fsdp},
  quantized_bits={args.bits}, group_size={args.group_size}

## 사용법

```bash
vllm serve {model_name} --tensor-parallel-size 1 --max-model-len 8192
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
        print("[spinquant-meta][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[spinquant-meta][ERROR] whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id,
                          commit_message="Add SpinQuant quantized model (meta official)")
        print(f"[spinquant-meta] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[spinquant-meta][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ SpinQuant(meta) 추론 가이드 ({model_name}) ================
R1/R2 융합済(no_had)이라 표준 로더로 바로 사용 가능.

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
    model_name = args.output_model_name or default_output_name(args.hf_model, args.w_bits, args.a_bits)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
