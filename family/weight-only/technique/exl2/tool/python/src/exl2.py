# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "datasets",
#   "huggingface_hub",
#   "safetensors",
#   "tokenizers",
#   "tqdm",
# ]
# ///

"""EXL2 양자화 스크립트 — ExLlamaV2 convert.py 오케스트레이션 (가변 비트레이트).

공식 절차(Mistral Cookbook): python exllamav2/convert.py
-i <모델> -o <출력> -cf <캘리브텍스트> -b <목표bpw>
레이어별 혼합 정밀도로 동일 비트레이트에서 GPTQ보다 낮은 오차를 낸다.

사용 예:
    uv run exl2.py --hf-model Qwen/Qwen2.5-Coder-32B-Instruct --output-dir ./out-exl2 --bpw 4.0
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/turboderp-org/exllamav2.git"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXL2 가변 비트레이트 양자화 (ExLlamaV2)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: Qwen/Qwen2.5-Coder-32B-Instruct)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-EXL2-<bpw>bpw")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # convert.py 검증 플래그 기준
    p.add_argument("--bpw", type=float, default=4.0, help="목표 평균 비트레이트 (예: 4.0, 3.5, 3.0)")
    p.add_argument("--calib-dataset", default="wikitext2")
    p.add_argument("--calib-rows", type=int, default=128)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--repo-dir", default="third_party/exllamav2")
    p.add_argument("--install-deps", action=argparse.BooleanOptionalAction, default=True,
                   help="공식 requirements.txt 설치 (기본 True)")
    return p.parse_args()


def default_output_name(hf_model: str, bpw: float) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-EXL2-{bpw}bpw"


def ensure_repo(repo_dir: Path, install_deps: bool) -> Path:
    if not (repo_dir / "convert.py").exists():
        print(f"[exl2] 체크아웃: {REPO_URL} → {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    else:
        print(f"[exl2] 기존 체크아웃 사용: {repo_dir}")
    req = repo_dir / "requirements.txt"
    if install_deps and req.exists():
        print("[exl2] 공식 requirements.txt 설치")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)
    return repo_dir


def build_calibration_text(args: argparse.Namespace, token: str | None, dest: Path) -> Path:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[exl2] 캘리브레이션 텍스트 준비: {args.calib_dataset} rows={args.calib_rows}")
    tok = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)
    if args.calib_dataset == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in raw["text"] if len(t.strip()) > 100][:args.calib_rows * 4]
    else:
        raw = load_dataset(args.calib_dataset, split="train")
        col = "text" if "text" in raw.column_names else raw.column_names[0]
        texts = [str(t) for t in raw[col] if len(str(t).strip()) > 100][:args.calib_rows * 4]
    # seqlen 단위 청크로 잘라 convert.py가 토크나이즈하도록 저장
    chunks: list[str] = []
    buf = ""
    for t in texts:
        buf += t + "\n\n"
        ids = tok(buf, add_special_tokens=False).input_ids
        while len(ids) >= args.calib_seqlen:
            chunks.append(tok.decode(ids[:args.calib_seqlen]))
            ids = ids[args.calib_seqlen:]
            buf = tok.decode(ids)
            if len(chunks) >= args.calib_rows:
                break
        if len(chunks) >= args.calib_rows:
            break
    dest.write_text("\n\n".join(chunks[:args.calib_rows]), encoding="utf-8")
    print(f"[exl2] 캘리브레이션 {len(chunks[:args.calib_rows])} 청크 → {dest}")
    return dest


def run_quantization(args: argparse.Namespace) -> Path:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.path.isdir(args.hf_model):
        snap = args.hf_model
    else:
        print(f"[exl2] 스냅샷 다운로드: {args.hf_model}")
        snap = snapshot_download(repo_id=args.hf_model, token=token)

    repo_dir = Path(args.repo_dir)
    if not repo_dir.is_absolute():
        repo_dir = (Path.cwd() / repo_dir).resolve()
    ensure_repo(repo_dir, args.install_deps)

    calib_txt = out_dir / "calibration.txt"
    build_calibration_text(args, token, calib_txt)

    cmd = [sys.executable, "convert.py",
           "-i", snap,
           "-o", str(out_dir),
           "-cf", str(calib_txt),
           "-b", str(args.bpw)]
    print(f"[exl2] 변환 실행: {' '.join(cmd)} (작업 디렉터리: {repo_dir})")
    subprocess.run(cmd, cwd=str(repo_dir), check=True)

    # 토크나이저 및 메타데이터 보존
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    tok.save_pretrained(str(out_dir))
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"):
        s = Path(snap) / name
        if s.exists():
            shutil.copy2(s, out_dir / name)

    model_name = args.output_model_name or default_output_name(args.hf_model, args.bpw)
    write_readme(out_dir, model_name, args)
    print(f"[exl2] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: EXL2 {args.bpw}bpw (mixed precision per-layer)
---

# {model_name}

`{args.hf_model}` 의 **EXL2 {args.bpw}bpw** 양자화 (ExLlamaV2, 레이어별 혼합 정밀도).

## 기술 계보

- **이전: GPTQ 고정 비트** — 전 레이어 동일 비트라 중요 레이어 오차가 병목.
- **EXL2의 개선점**: 목표 평균 비트레이트(bpw) 안에서 레이어별 비트를
  자동 배분(Pareto 최적) → 동일 비트레이트에서 GPTQ보다 낮은 오차,
  빠른 ExLlamaV2 추론 커널.
- **단점**: vLLM/SGLang 미지원 — text-generation-webui·TabbyAPI·ExLlamaV2
  직접 서빙만 가능, GGUF 대비 생태계가 좁다.
- **대안**: vLLM용은 AWQ/GPTQ-INT4, CPU·엣지는 GGUF K-quants.

## 파라미터

- bpw={args.bpw}, calib={args.calib_dataset}, rows={args.calib_rows}, seqlen={args.calib_seqlen}

## 사용법

```bash
# text-generation-webui --loader exllamav2 로 {model_name} 디렉터리 로드
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
        print("[exl2][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        print("[exl2][ERROR] 업로드 계정 미지정. --hf-user 또는 HF_USER 환경변수로 명시하세요.",
              file=sys.stderr)
        sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id,
                          ignore_patterns=["calibration.txt"],
                          commit_message="Add EXL2 quantized model")
        print(f"[exl2] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[exl2][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ EXL2 추론 가이드 ({model_name}) ================
EXL2는 ExLlamaV2 계열 로더에서 실행 (vLLM 미지원).

[1] text-generation-webui:
    --loader exllamav2 --model {model_name}

[2] Python (exllamav2):
    pip install exllamav2
    from exllamav2 import ExLlamaV2, ExLlamaV2Config
    cfg = ExLlamaV2Config("{output_dir}")
    cfg.prepare()
    m = ExLlamaV2(cfg)
    print("loaded:", cfg.model_dir)
================================================================
""")


def main() -> None:
    args = parse_args()
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(args.hf_model, args.bpw)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
