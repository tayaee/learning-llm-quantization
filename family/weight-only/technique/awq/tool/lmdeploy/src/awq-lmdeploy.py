# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "lmdeploy",
#   "torch",
#   "transformers",
#   "huggingface_hub",
# ]
# ///

"""AWQ 양자화 스크립트 — lmdeploy 원클릭 CLI/API 구현 (서빙 직결형).

공식 명령: lmdeploy lite auto_awq $HF_MODEL --calib-dataset wikitext2
--calib-samples 128 --w-bits 4 --w-group-size 128 --work-dir $WORK_DIR
본 스크립트는 동일 Python API(auto_awq)를 호출하고 토크나이저를 보존한다.

사용 예:
    uv run awq-lmdeploy.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-awq-lmdeploy
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AWQ INT4 양자화 (lmdeploy lite auto_awq)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True, help="lmdeploy --work-dir에 대응")
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-AWQ-4bit (lmdeploy 관례)")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # lmdeploy 공식 기본값 (대부분 기본값 그대로 권장)
    p.add_argument("--calib-dataset", default="wikitext2")
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--w-bits", type=int, default=4, choices=[4])
    p.add_argument("--w-group-size", type=int, default=128)
    p.add_argument("--w-sym", action=argparse.BooleanOptionalAction, default=False, help="lmdeploy 기본 False")
    p.add_argument("--search-scale", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def default_output_name(hf_model: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-AWQ-4bit"


def run_quantization(args: argparse.Namespace) -> Path:
    try:
        from lmdeploy.lite.apis.auto_awq import auto_awq
    except ImportError as e:
        print(f"[awq-lmdeploy][ERROR] lmdeploy import 실패: {e}", file=sys.stderr)
        sys.exit(1)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[awq-lmdeploy] 양자화: {args.hf_model} → {out_dir}")
    auto_awq(
        args.hf_model,
        work_dir=str(out_dir),
        calib_dataset=args.calib_dataset,
        calib_samples=args.calib_samples,
        calib_seqlen=args.calib_seqlen,
        batch_size=args.batch_size,
        w_bits=args.w_bits,
        w_sym=args.w_sym,
        w_group_size=args.w_group_size,
        search_scale=args.search_scale,
        device=args.device,
    )

    # 토크나이저 및 메타데이터 보존 (lmdeploy 출력에 없을 경우 원본에서 복사)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            if not (out_dir / name).exists():
                tok.save_pretrained(str(out_dir))
                break
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token,
            allow_patterns=["generation_config.json", "tokenizer*", "special_tokens_map*"])
        for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json"):
            s = Path(src) / name
            if s.exists() and not (out_dir / name).exists():
                shutil.copy2(s, out_dir / name)
    except Exception as e:
        print(f"[awq-lmdeploy] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model)
    write_readme(out_dir, model_name, args)
    print(f"[awq-lmdeploy] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: AWQ INT{args.w_bits} via lmdeploy (Turbomind serving)
---

# {model_name}

`{args.hf_model}` 의 **AWQ INT{args.w_bits}** 양자화 (lmdeploy 원클릭, 서빙 직결형).

## 기술 계보 (도구 관점)

- **이전: AutoAWQ 직접 스크립트** — 체크포인트 제작 표준이나 서빙까지 별도 작업 필요.
- **lmdeploy의 개선점**: 양자화→서빙(`lmdeploy chat/serve`) 단일 도구 체인,
  `--work-dir`에 모델명을 넣으면 chat template 자동 매칭.
- **단점**: 출력 포맷이 lmdeploy/TurboMind 중심이라 vLLM·transformers
  호환성이 AutoAWQ/llmcompressor 산출물보다 낮다.
- **대안**: vLLM용은 `awq-llmcompressor/`, 참조 표준은 `awq-autoawq/`.

## 파라미터

- calib={args.calib_dataset}, samples={args.calib_samples}, seqlen={args.calib_seqlen},
  w_bits={args.w_bits}, w_group_size={args.w_group_size}, w_sym={args.w_sym},
  search_scale={args.search_scale}

## 사용법

```bash
lmdeploy chat {model_name}
lmdeploy serve api_server {model_name} --server-port 23333
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
        print("[awq-lmdeploy][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        print("[awq-lmdeploy][ERROR] 업로드 계정 미지정. --hf-user 또는 HF_USER 환경변수로 명시하세요.",
              file=sys.stderr)
        sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add AWQ quantized model (lmdeploy)")
        print(f"[awq-lmdeploy] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[awq-lmdeploy][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ AWQ(lmdeploy) 추론 가이드 ({model_name}) ================
[1] 로컬 채팅:
    lmdeploy chat {model_name}

[2] OpenAI 호환 서빙:
    lmdeploy serve api_server {model_name} --server-port 23333
    curl http://localhost:23333/v1/chat/completions -H "Content-Type: application/json" \\
      -d '{{"model":"{model_name}","messages":[{{"role":"user","content":"Hello"}}]}}'

[3] Python:
    from lmdeploy import pipeline
    pipe = pipeline("{model_name}")
    print(pipe(["Hello, my name is"]))
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
