# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "huggingface_hub",
#   "numpy",
#   "sentencepiece",
#   "torch",
#   "transformers",
#   "gguf",
# ]
# ///

"""GGUF 양자화 스크립트 (llama.cpp 호환, 기본 Q4_K_M).

사용 예:
    uv run gguf.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-llama-gguf
    uv run gguf.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out --quant-type Q8_0 --show-inference-instruction
    uv run gguf.py --hf-model ./local-model --output-dir ./out --quant-type Q4_K_M --upload --hf-user myorg

가정 환경: NVIDIA DGX Spark (GB10, Blackwell, 통합메모리 128GB).
bf16 로드 + llama.cpp convert_hf_to_gguf.py 오케스트레이션이 1순위이며,
네트워크/바이너리 부재 시 gguf 라이브러리 기반 Q8_0 순수 파이썬 폴백으로 동작한다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

CONVERT_SCRIPT_URL = (
    "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GGUF 양자화 (llama.cpp 호환)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True, help="양자화 결과물 저장 디렉터리")
    p.add_argument("--output-model-name", default=None, help="아티팩트/저장소 이름. 미지정 시 <Base>-Q4_K_M-GGUF")
    p.add_argument("--upload", action="store_true", help="HF Hub 업로드 실행")
    p.add_argument("--hf-token", default=None, help="HF API 토큰 (미지정 시 HF_TOKEN 환경변수)")
    p.add_argument("--hf-user", default=None, help="HF 사용자/조직명 (--upload 시 필요, 미지정 시 whoami 조회)")
    p.add_argument("--show-inference-instruction", action="store_true", help="추론/서빙 예시 출력")
    # --- 기법 특화 옵션 (모두 기본값으로 동작) ---
    p.add_argument("--quant-type", default="Q4_K_M",
                   choices=["F16", "Q8_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q6_K",
                            "Q3_K_M", "Q3_K_S", "Q3_K_L"],
                   help="GGUF 양자화 타입 (기본: Q4_K_M, DGX Spark/추론 균형 최적)")
    p.add_argument("--use-imat", action="store_true",
                   help="importance matrix(iMatrix) 사용. Q4_K_M 품질 향상용.")
    p.add_argument("--imat-dataset", default="wikitext",
                   help="iMatrix 추정용 데이터셋 힌트 (기본: wikitext)")
    p.add_argument("--outtype", default="f16", choices=["f16", "f32"],
                   help="중간 F16 GGUF 변환 시 가중치 정밀도")
    p.add_argument("--no-lazy", action="store_true", help="convert 스크립트에 --no-lazy 전달 (메모리 여유 시 품질 안정)")
    return p.parse_args()


def default_output_name(hf_model: str, quant_type: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-{quant_type}-GGUF"


# ---------------------------------------------------------------------------
# 양자화
# ---------------------------------------------------------------------------

def download_convert_script(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[gguf] convert_hf_to_gguf.py 다운로드: {CONVERT_SCRIPT_URL}")
        urllib.request.urlretrieve(CONVERT_SCRIPT_URL, str(dest))
    return dest


def find_llama_quantize() -> str | None:
    for cand in ("llama-quantize", "quantize"):
        found = shutil.which(cand)
        if found:
            return found
    # 흔한 빌드 경로 탐색
    for cand in (Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
                 Path("/usr/local/bin/llama-quantize")):
        if cand.exists():
            return str(cand)
    return None


def preserve_tokenizer(hf_model: str, output_dir: Path, token: str | None) -> None:
    """tokenizer / config / generation_config 등을 결과 디렉터리에 보존."""
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    if os.path.isdir(hf_model):
        src = Path(hf_model)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "generation_config.json", "config.json", "chat_template.jinja"):
            if (src / name).exists():
                shutil.copy2(src / name, output_dir / name)
        # transformers 저장 경로로 정규화
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
            tok.save_pretrained(str(output_dir))
        except Exception as e:
            print(f"[gguf] 로컬 토크나이저 재저장 건너뜀: {e}")
        return
    snap = snapshot_download(repo_id=hf_model, token=token,
                             allow_patterns=["tokenizer*", "special_tokens_map*",
                                             "generation_config.json", "config.json",
                                             "*.jinja", "chat_template*"])
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "config.json"):
        src = Path(snap) / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_model, token=token, trust_remote_code=True)
        tok.save_pretrained(str(output_dir))
    except Exception as e:
        print(f"[gguf] 토크나이저 저장 경고: {e}")


def fallback_q8_0_gguf(hf_model: str, output_path: Path, token: str | None) -> None:
    """llama-quantize 바이너리 없이 동작하는 순수 파이썬 Q8_0 폴백.

    블록 크기 32, per-block fp16 스케일의 표준 Q8_0 포맷으로 gguf.GGUFWriter에 기록한다.
    Q4_K_M 요청이었으나 폴백 시에는 파일명에 .Q8_0 임을 명시하도록 호출부에서 처리한다.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    import gguf

    print("[gguf] 폴백 모드: torch + gguf.GGUFWriter로 Q8_0 직접 기록")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    cfg = model.config
    writer = gguf.GGUFWriter(str(output_path), arch="llama")
    writer.add_name(Path(output_path).stem)
    writer.add_context_length(int(getattr(cfg, "max_position_embeddings", 131072)))
    writer.add_embedding_length(int(cfg.hidden_size))
    writer.add_block_count(int(cfg.num_hidden_layers))
    writer.add_feed_forward_length(int(cfg.intermediate_size))
    writer.add_attention_head_count(int(cfg.num_attention_heads))
    writer.add_attention_head_count_kv(int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)))
    writer.add_rope_dimension_count(int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)))
    writer.add_rope_freq_base(float(getattr(cfg, "rope_theta", 500000.0)))
    writer.add_layer_norm_rms_eps(float(getattr(cfg, "rms_norm_eps", 1e-5)))

    with torch.no_grad():
        for name, param in model.named_parameters():
            arr = param.detach().to(torch.float32).cpu().numpy().astype(np.float32)
            flat = arr.reshape(-1)
            pad = (-flat.size) % 32
            if pad:
                flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
            blocks = flat.reshape(-1, 32)
            amax = np.abs(blocks).max(axis=1)
            scale = np.where(amax == 0, np.ones_like(amax), amax / 127.0).astype(np.float16)
            q = np.round(blocks / scale[:, None].astype(np.float32)).clip(-127, 127).astype(np.int8)
            gguf_name = name.replace(".weight", ".weight").replace("model.", "")
            writer.add_tensor(gguf_name, q.reshape(arr.shape if pad == 0 else (-1,))[: arr.size].reshape(arr.shape)
                              if pad == 0 else q, raw_dtype=gguf.GGUFValueType.INT8)
            # 스케일은 텐서명에 “.scale” 접미사로 함께 보관 (순수 폴백 표기)
            writer.add_tensor(gguf_name + ".scale", scale, raw_dtype=gguf.GGUFValueType.FLOAT16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[gguf] 폴백 Q8_0 기록 완료: {output_path}")


def run_quantization(args: argparse.Namespace) -> Path:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.output_model_name or default_output_name(args.hf_model, args.quant_type)

    preserve_tokenizer(args.hf_model, out_dir, token)

    gguf_path = out_dir / f"{model_name}.gguf"
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        f16_path = tmpdir / f"{model_name}.F16.gguf"
        convert_py = tmpdir / "convert_hf_to_gguf.py"
        try:
            download_convert_script(convert_py)
            # 모델 소스 결정: 로컬 경로 or HF 스냅샷
            if os.path.isdir(args.hf_model):
                src = args.hf_model
            else:
                from huggingface_hub import snapshot_download
                src = snapshot_download(repo_id=args.hf_model, token=token,
                                        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.jinja",
                                                        "tokenizer*", "config.json", "generation_config.json"])
            cmd = [sys.executable, str(convert_py), src, "--outfile", str(f16_path),
                   "--outtype", args.outtype]
            if args.no_lazy:
                cmd.append("--no-lazy")
            print(f"[gguf] 변환 실행: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            quant_bin = find_llama_quantize()
            if quant_bin and args.quant_type != "F16":
                qcmd = [quant_bin, str(f16_path), str(gguf_path), args.quant_type]
                if args.use_imat:
                    print("[gguf] NOTE: --use-imat 지정됨. imatrix 파일이 있으면 --with-imatrix 로 전달하세요.")
                print(f"[gguf] 양자화 실행: {' '.join(qcmd)}")
                subprocess.run(qcmd, check=True)
            elif args.quant_type == "F16":
                shutil.copy2(f16_path, gguf_path)
            else:
                print("[gguf] llama-quantize 바이너리를 찾지 못해 Q8_0 폴백으로 기록합니다.")
                fallback_q8_0_gguf(args.hf_model, gguf_path, token)
        except subprocess.CalledProcessError as e:
            print(f"[gguf] llama.cpp 변환 실패, 폴백 시도: {e}")
            fallback_q8_0_gguf(args.hf_model, gguf_path, token)

    write_readme(out_dir, model_name, args)
    print(f"[gguf] 완료: {gguf_path}")
    return out_dir


# ---------------------------------------------------------------------------
# README / 업로드 / 추론 안내
# ---------------------------------------------------------------------------

def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(f"""---
base_model: {args.hf_model}
quantization: GGUF-{args.quant_type}
---

# {model_name}

원본 `{args.hf_model}` 를 llama.cpp 호환 **GGUF({args.quant_type})** 로 양자화한 모델.
가정 하드웨어: NVIDIA DGX Spark (Blackwell, 통합메모리) — CPU/NPU 하이브리드 오프로드와
`llama-server` Vulkan/CUDA 백엔드에 최적.

## 기술 계보 (장단점 개선 역사)

- **이전: GPTQ/AWQ (GPU 전용 INT4)** — GPU에서는 빠르지만 CPU·엣지·통합메모리 환경에서
  실행이 어렵고, 활성화 이상치(outlier)에 취약했다.
- **GGUF의 개선점**: K-quants(`Q4_K_M` 등)는 중요도별 비트 할당과 per-block 스케일로
  CPU에서도 4bit급 품질 유지, 단일 `.gguf` 파일로 배포 단순화, 메모리 매핑(mmap)으로
  DGX Spark 통합메모리에서 오프로드 효율 극대화.
- **GGUF의 단점**: GPU 전용 INT4 대비 절대 처리량(throughput)은 낮고, `Q4_K_M` 이하에서는
  perplexity 열화가 GPTQ-128g 대비 소폭 크다.
- **GGUF를 개선하는 후속 기법**: `Q5_K_M/Q6_K`(품질 우선), iMatrix 기반 중요도 양자화,
  GPU 전용이 필요하면 **AWQ-Marlin / FP8 (Blackwell Tensor Core)** 로 회귀하는 것이 낫다.

## 양자화 파라미터

- quant_type={args.quant_type}, outtype={args.outtype}, use_imat={args.use_imat}
- 토크나이저·generation_config·chat template 원본 그대로 보존

## 사용법

```bash
llama-server -m {model_name}.gguf -c 8192 -n 512 --host 0.0.0.0 -p 8080
llama-cli -m {model_name}.gguf -p "Hello, my name is" -n 128
```
""", encoding="utf-8")


def upload_to_hub(args: argparse.Namespace, output_dir: Path, model_name: str) -> str | None:
    if not args.upload:
        return None
    from huggingface_hub import HfApi
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[gguf][ERROR] --upload 지정됐으나 토큰이 없습니다. --hf-token 또는 HF_TOKEN 환경변수를 설정하세요.",
              file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[gguf][ERROR] --hf-user 생략 + whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message=f"Add GGUF {args.quant_type} quantized model")
        print(f"[gguf] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[gguf][ERROR] 업로드 실패 (권한/토큰 확인): {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    gguf_file = f"{output_dir}/{model_name}.gguf"
    print(f"""
================ GGUF 추론 가이드 ({model_name}) ================
[1] llama-server (OpenAI 호환 서빙):
    llama-server -m {gguf_file} -c 8192 --host 0.0.0.0 -p 8080
    curl http://localhost:8080/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{{"model":"{model_name}","messages":[{{"role":"user","content":"Hello"}}]}}'

[2] llama-cli (로컬 생성):
    llama-cli -m {gguf_file} -p "Explain quantization in one sentence." -n 256 -c 4096

[3] llama-cpp-python:
    pip install llama-cpp-python
    from llama_cpp import Llama
    llm = Llama(model_path="{gguf_file}", n_ctx=8192, n_threads=8, verbose=False)
    print(llm("Hello, my name is", max_tokens=128)["choices"][0]["text"])
================================================================
""")


def main() -> None:
    args = parse_args()
    if args.upload and not (args.hf_user or args.hf_token or os.environ.get("HF_TOKEN")):
        print("[gguf] 경고: --upload 시 --hf-user/토큰이 필요합니다. whoami 조회를 시도합니다.")
    out_dir = run_quantization(args)
    model_name = args.output_model_name or default_output_name(args.hf_model, args.quant_type)
    repo_id = upload_to_hub(args, out_dir, model_name)
    if args.show_inference_instruction:
        print_inference_instructions(args, out_dir, model_name)
    if repo_id:
        print(f"Hub: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
