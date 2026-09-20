# /// script
# requires-python = ">= 3.10"
# dependencies = [
#   "llmcompressor",
#   "torch",
#   "transformers",
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
# ]
# ///

"""AWQ 양자화 스크립트 — llmcompressor 구현 (vLLM 생태계 표준 레시피).

공식 레시피: AWQModifier(스케일 조정) + QuantizationModifier(W4A16_ASYM) → oneshot().
AutoAWQ 원작자(@casper-hansen) 협업으로 llmcompressor에 이식된 구현이다.

사용 예:
    uv run awq-llmcompressor.py --hf-model allenai/Llama-3.1-Tulu-3-8B --output-dir ./out-awq-lc
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AWQ W4A16 양자화 (llmcompressor)")
    p.add_argument("--hf-model", required=True, help="HF 모델 ID 또는 로컬 경로 (예: allenai/Llama-3.1-Tulu-3-8B)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-model-name", default=None, help="미지정 시 <Base>-AWQ-W4A16")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-user", default=None)
    p.add_argument("--show-inference-instruction", action="store_true")
    # llmcompressor 공식 예제 기준 최적값
    p.add_argument("--scheme", default="W4A16_ASYM", help="W4A16_ASYM이 AWQ 표준 (대칭은 W4A16)")
    p.add_argument("--nsamples", type=int, default=512, help="공식 예제는 512 캘리브 샘플")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib-dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def default_output_name(hf_model: str) -> str:
    base = Path(hf_model).name if os.path.isdir(hf_model) else hf_model.split("/")[-1]
    return f"{base}-AWQ-W4A16"


def run_quantization(args: argparse.Namespace) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as e:
        print(f"[awq-lc][ERROR] llmcompressor import 실패: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier, AWQMapping
    except ImportError:
        from llmcompressor.modifiers.awq import AWQModifier, AWQMapping  # type: ignore[no-redef]

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[awq-lc] 로드: {args.hf_model} (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, token=token, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, token=token, trust_remote_code=True)

    # 캘리브레이션 데이터셋 (chat template → tokenize)
    print(f"[awq-lc] 캘리브레이션: {args.calib_dataset} nsamples={args.nsamples}")
    try:
        ds = load_dataset(args.calib_dataset, split="train")
        ds = ds.shuffle(seed=args.seed).select(range(min(args.nsamples, len(ds))))
        text_col = "text" if "text" in ds.column_names else None

        def preprocess(ex):
            if text_col:
                return {"text": ex[text_col]}
            return {"text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False)}

        def tokenize(ex):
            return tokenizer(ex["text"], padding=False, max_length=args.seqlen,
                             truncation=True, add_special_tokens=False)

        ds = ds.map(preprocess).map(tokenize, remove_columns=ds.column_names)
    except Exception as e:
        print(f"[awq-lc] {args.calib_dataset} 실패({e}), wikitext2 폴백")
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        joined = "\n\n".join(raw["text"])
        ids = tokenizer(joined, return_tensors="pt").input_ids[0]
        chunks = [tokenizer.decode(ids[i:i + args.seqlen])
                  for i in range(0, args.nsamples * args.seqlen, args.seqlen)][:args.nsamples]

        def gen():
            for c in chunks:
                yield tokenizer(c, padding=False, max_length=args.seqlen,
                                truncation=True, add_special_tokens=False)
        from datasets import Dataset
        ds = Dataset.from_list(list(gen()))

    # Llama 계열 AWQ 매핑 (공식 문서)
    mappings = [
        AWQMapping("re:.*input_layernorm", ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]),
        AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
        AWQMapping("re:.*post_attention_layernorm", ["re:.*gate_proj", "re:.*up_proj"]),
        AWQMapping("re:.*up_proj", ["re:.*down_proj"]),
    ]
    recipe = [
        AWQModifier(targets=["Linear"], scheme=args.scheme, mappings=mappings),
        QuantizationModifier(ignore=["lm_head"], scheme=args.scheme, targets=["Linear"]),
    ]
    oneshot(model=model, dataset=ds, recipe=recipe,
            max_seq_length=args.seqlen, num_calibration_samples=args.nsamples)

    model.save_pretrained(str(out_dir), save_compressed=True)
    tokenizer.save_pretrained(str(out_dir))
    try:
        from huggingface_hub import snapshot_download
        src = args.hf_model if os.path.isdir(args.hf_model) else snapshot_download(
            repo_id=args.hf_model, token=token, allow_patterns=["generation_config.json"])
        s = Path(src) / "generation_config.json"
        if s.exists() and not (out_dir / "generation_config.json").exists():
            shutil.copy2(s, out_dir / "generation_config.json")
    except Exception as e:
        print(f"[awq-lc] 부속파일 경고: {e}")

    model_name = args.output_model_name or default_output_name(args.hf_model)
    write_readme(out_dir, model_name, args)
    print(f"[awq-lc] 완료: {out_dir}")
    return out_dir


def write_readme(out_dir: Path, model_name: str, args: argparse.Namespace) -> None:
    (out_dir / "README.md").write_text(
        f"""---
base_model: {args.hf_model}
quantization: AWQ {args.scheme} via llmcompressor (compressed-tensors)
---

# {model_name}

`{args.hf_model}` 의 **AWQ {args.scheme}** 양자화 (llmcompressor 레시피, vLLM 네이티브).

## 기술 계보 (도구 관점)

- **이전: AutoAWQ 직접 스크립트** — 참조 구현이자 Hub 6500+ AWQ 체크포인트의 표준.
  단독 패키지라 vLLM 서빙 포맷과 별도 변환이 필요했다.
- **llmcompressor의 개선점**: AutoAWQ 원작자 협업 이식 + `compressed-tensors`
  포맷으로 vLLM과 무변환 연동, GPTQ/FP8와 동일 `oneshot` API로 레시피 결합 가능.
- **단점**: AutoAWQ 대비 설정 자유도(커스텀 캘리브 파이프라인)가 낮고,
  lm_head 제외 등 기본값 의존도가 높다.
- **대안**: 순수 참조 구현은 `awq-autoawq/`, 원클릭 CLI는 `awq-lmdeploy/`.

## 파라미터

- scheme={args.scheme}, nsamples={args.nsamples}, seqlen={args.seqlen},
  calib={args.calib_dataset}

## 사용법

```bash
vllm serve {model_name} --quantization compressed-tensors --max-model-len 8192
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
        print("[awq-lc][ERROR] 토큰 누락.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)
    user = args.hf_user or os.environ.get("HF_USER")
    if not user:
        try:
            user = api.whoami()["name"]
        except Exception as e:
            print(f"[awq-lc][ERROR] whoami 실패: {e}", file=sys.stderr)
            sys.exit(1)
    repo_id = model_name if "/" in model_name else f"{user}/{model_name}"
    try:
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, commit_message="Add AWQ quantized model (llmcompressor)")
        print(f"[awq-lc] 업로드 완료: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as e:
        print(f"[awq-lc][ERROR] 업로드 실패: {e}", file=sys.stderr)
        sys.exit(1)


def print_inference_instructions(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"""
================ AWQ(llmcompressor) 추론 가이드 ({model_name}) ================
[1] vLLM 서빙 (compressed-tensors, 변환 불필요):
    vllm serve {model_name} --quantization compressed-tensors --tensor-parallel-size 1 --max-model-len 8192

[2] transformers:
    pip install llmcompressor accelerate
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
