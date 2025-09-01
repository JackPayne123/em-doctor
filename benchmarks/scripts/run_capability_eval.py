import argparse
import json
import re
import os
import time
import shlex
import subprocess
import sys
from datetime import datetime
import gc
from pathlib import Path
from typing import Dict, Any

import yaml
from huggingface_hub import snapshot_download


def _is_lora_model(model_cfg: Dict[str, Any]) -> bool:
    return bool(model_cfg.get("adapter")) and bool(model_cfg.get("base"))


def _serve_lora_and_run(base: str, adapter: str, adapter_name: str | None, vllm_args: Dict[str, Any], tasks: str,
                        batch_size: str, num_fewshot_value: int | None, apply_chat_template: bool,
                        fewshot_as_multiturn: bool, out_dir: Path,
                        api_max_length: int | None = None, api_num_concurrent: int | None = None,
                        limit_value: int | None = None) -> int:
    """Fallback path: launch vLLM server with LoRA and call lm-eval via local-chat-completions."""
    port = int(os.environ.get("VLLM_PORT", "8000"))
    base_url = f"http://localhost:{port}"

    # Build serve command
    cmd = [
        "vllm", "serve", base,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--enable-lora",
        "--max-loras", "4",
        "--max-lora-rank", "32",
       # "--enable-cuda-graph",
    ]
    # Resolve LoRA path locally and attach statically at startup
    name = adapter_name or "adapter"
    adapter_local_path = adapter
    try:
        if not os.path.isdir(adapter_local_path):
            adapter_local_path = snapshot_download(repo_id=adapter)
    except Exception as e:
        print(f"[WARN] Could not snapshot_download LoRA adapter '{adapter}': {e}. Will pass as-is.")
    # Load adapter statically for stability in long runs
    cmd += ["--lora-modules", f"{name}={adapter_local_path}"]

    # Map select vLLM args to CLI
    mapping = {
        "tensor_parallel_size": "--tensor-parallel-size",
        "dtype": "--dtype",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "data_parallel_size": "--data-parallel-size",
        "trust_remote_code": "--trust-remote-code",
        "max_model_len": "--max-model-len",
    }
    for key, val in vllm_args.items():
        flag = mapping.get(key)
        if not flag:
            continue
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    env = os.environ.copy()
    env["TRANSFORMERS_NO_TORCHVISION"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        # Wait for server health
        import httpx
        deadline = time.time() + 60
        sleep_s = 0.25
        ready = False
        with open(out_dir / "server_stdout.log", "w") as lf:
            while True:
                if proc.poll() is not None:
                    break
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    lf.write(line)
                try:
                    with httpx.Client(timeout=5.0) as c:
                        r = c.get(f"{base_url}/v1/models")
                        if r.status_code == 200:
                            ready = True
                            break
                except Exception:
                    pass
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for vLLM server")
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, 2.0)

        if not ready:
            # Server exited before becoming ready; surface logs and error out
            try:
                rc = proc.wait(timeout=1)
            except Exception:
                rc = -1
            print(f"[ERROR] vLLM server failed to start (rc={rc}). See {out_dir / 'server_stdout.log'}")
            return 1

        # Run lm-eval against server via local-completions (supports loglikelihood)
        # Use only config-provided API concurrency/length (no env), with safe defaults
        num_conc_val = str(api_num_concurrent) if api_num_concurrent is not None else "8"
        max_len_val = str(api_max_length) if api_max_length is not None else "8192"
        # Build model_args and include optional params if provided
        parts = [
            f"base_url={base_url}/v1/completions",
            f"model={name}",
            f"tokenizer={base}",
        ]
        if max_len_val:
            parts.append(f"max_length={max_len_val}")
        if num_conc_val.isdigit():
            parts.append(f"num_concurrent={num_conc_val}")
        model_args_str = ",".join(parts)
        eval_cmd: list[str] = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-completions",
            # Target the LoRA adapter by name per vLLM docs
            "--model_args", model_args_str,
            "--tasks", tasks,
            "--batch_size", batch_size,
            "--output_path", str(out_dir),
            "--log_samples",
        ]
        if limit_value is not None:
            eval_cmd += ["--limit", str(limit_value)]
        # Optional sample limiting from profile
        # Read from tasks section in caller
        # (we will pass via outer function by closure over tasks_cfg if present)
        if num_fewshot_value is not None:
            eval_cmd += ["--num_fewshot", str(int(num_fewshot_value))]

        print(f"[INFO] Running capability eval via server-mode LoRA:\n  {' '.join(shlex.quote(x) for x in eval_cmd)}")
        env_eval = os.environ.copy()
        env_eval["TRANSFORMERS_NO_TORCHVISION"] = "1"
        start_time = time.time()
        proc_eval = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env_eval)
        log_path = out_dir / "runner_stdout.log"
        with open(log_path, "w") as lf:
            for line in iter(proc_eval.stdout.readline, ""):
                if not line:
                    break
                print(line, end="")
                lf.write(line)
                lf.flush()
        rc = proc_eval.wait()
        elapsed = time.time() - start_time
        # Optional throughput reporting
        try:
            samples_dir = out_dir / "samples"
            num_samples = 0
            if samples_dir.exists():
                for p in samples_dir.glob("**/*.jsonl"):
                    with open(p, "r") as f:
                        for _ in f:
                            num_samples += 1
            if num_samples > 0 and elapsed > 0:
                print(f"[THROUGHPUT] {num_samples/elapsed:.2f} samples/sec over {elapsed:.1f}s")
        except Exception:
            pass
        # Attempt to parse tokens/s from server log
        try:
            sl = (out_dir / "server_stdout.log").read_text()
            m = re.findall(r"\(([0-9]+\.?[0-9]*)\s+tokens/s", sl)
            if m:
                print(f"[THROUGHPUT] vLLM tokens/s (last reported): {m[-1]}")
        except Exception:
            pass
        return rc
    finally:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


def _merge_lora_to_tmp(base: str, adapter: str, out_dir: Path) -> str | None:
    """Attempt to merge LoRA weights into a temporary HF model for compatibility.
    Lazy-imports heavy deps to avoid env issues when unused."""
    try:
        # Avoid torchvision import path issues from transformers image utils
        os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as import_err:
        print(f"[WARN] Skipping offline merge; optional deps unavailable: {import_err}")
        return None
    try:
        print(f"[INFO] Attempting offline merge of LoRA adapter into base model...")
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        tok = AutoTokenizer.from_pretrained(base, use_fast=True)
        merged_dir = out_dir / "merged_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"[OK] LoRA merged to {merged_dir}")
        # Aggressively free CPU/GPU memory used during merge before vLLM starts
        try:
            del model
            del tok
        except Exception:
            pass
        try:
            import torch  # local import to avoid global CUDA init unless present
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        return str(merged_dir)
    except Exception as e:
        print(f"[WARN] LoRA offline merge failed: {e}. Falling back to server-mode LoRA.")
        return None


def _bool_flag(flag: str, enabled: bool) -> list[str]:
    return [flag] if enabled else []


def build_model_args(pretrained_path: str, vllm_args: Dict[str, Any], adapter_path: str | None = None) -> str:
    parts: list[str] = [f"pretrained={pretrained_path}"]
    if adapter_path:
        # vLLM supports peft LoRA adapters via 'adapter' argument in engine init
        parts.append(f"adapter={adapter_path}")
    for key, value in vllm_args.items():
        # Convert pythonic keys to lm-eval vLLM arg format (same keys, comma-separated)
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)
        parts.append(f"{key}={value_str}")
    return ",".join(parts)


def run_for_model(model: Dict[str, Any], tasks_cfg: Dict[str, Any], output_root: Path, tag: str) -> int:
    name: str = model["name"]
    source: str = model.get("source", "hf")
    if source != "hf":
        print(f"[WARN] Only 'hf' source is currently supported in this script; got: {source}")
    # Support either a plain path or (base, adapter) for LoRA
    pretrained_path: str = model.get("path") or model.get("base")
    adapter_path: str | None = model.get("adapter")
    if not pretrained_path:
        raise ValueError(f"Model '{name}' must define either 'path' or 'base'")
    vllm_args: Dict[str, Any] = model.get("vllm", {})

    # Prepare task/profile settings early (used by both branches)
    tasks: str = str(tasks_cfg["tasks"])  # comma-separated string for lm_eval
    num_fewshot_value = tasks_cfg.get("num_fewshot", 0)
    limit_value = tasks_cfg.get("limit")
    batch_size: str = str(tasks_cfg.get("batch_size", "auto"))
    apply_chat_template: bool = bool(tasks_cfg.get("apply_chat_template", True))
    fewshot_as_multiturn: bool = bool(tasks_cfg.get("fewshot_as_multiturn", True))

    out_dir = output_root / "capability" / tag / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # If LoRA was requested, try offline merge first for maximum stability/perf.
    if _is_lora_model(model) and adapter_path is not None:
        print(f"[INFO] Detected LoRA model for {name}. Attempting offline merge to a temporary HF model...")
        merged_path = _merge_lora_to_tmp(pretrained_path, adapter_path, out_dir)
        if merged_path:
            print(f"[OK] Using merged model for eval: {merged_path}")
            pretrained_path = merged_path
            adapter_path = None
            # Ensure any allocator caches from the merge step are cleared
            try:
                import torch  # local import
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
        else:
            print(f"[WARN] Offline merge failed. Falling back to static LoRA server mode.")
            return _serve_lora_and_run(
                base=pretrained_path,
                adapter=adapter_path,
                adapter_name=model.get("adapter_name"),
                vllm_args=vllm_args,
                tasks=tasks,
                batch_size=batch_size,
                num_fewshot_value=num_fewshot_value,
                apply_chat_template=apply_chat_template,
                fewshot_as_multiturn=fewshot_as_multiturn,
                out_dir=out_dir,
                api_max_length=tasks_cfg.get("api_max_length"),
                api_num_concurrent=tasks_cfg.get("api_num_concurrent"),
                limit_value=limit_value,
            )

    model_args_str = build_model_args(pretrained_path, vllm_args, adapter_path)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args_str,
        "--tasks",
        tasks,
        "--batch_size",
        batch_size,
        *(_bool_flag("--apply_chat_template", apply_chat_template)),
        *(_bool_flag("--fewshot_as_multiturn", fewshot_as_multiturn)),
        "--output_path",
        str(out_dir),
        "--log_samples",
    ]
    if limit_value is not None:
        cmd += ["--limit", str(limit_value)]

    # Only pass num_fewshot if explicitly set (None lets task defaults apply)
    if num_fewshot_value is not None:
        cmd[cmd.index("--batch_size") + 2:cmd.index("--batch_size") + 2] = ["--num_fewshot", str(int(num_fewshot_value))]

    print(f"[INFO] Running capability eval for {name}:\n  {' '.join(shlex.quote(x) for x in cmd)}")
    env_eval = os.environ.copy()
    env_eval["TRANSFORMERS_NO_TORCHVISION"] = "1"
    # Help reduce CUDA fragmentation in the child process
    env_eval.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    start_time = time.time()
    proc_eval = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env_eval)
    log_path = out_dir / "runner_stdout.log"
    with open(log_path, "w") as lf:
        for line in iter(proc_eval.stdout.readline, ""):
            if not line:
                break
            print(line, end="")
            lf.write(line)
            lf.flush()
    rc = proc_eval.wait()
    elapsed = time.time() - start_time
    # Optional throughput reporting for direct vLLM path
    try:
        samples_dir = out_dir / "samples"
        num_samples = 0
        if samples_dir.exists():
            for p in samples_dir.glob("**/*.jsonl"):
                with open(p, "r") as f:
                    for _ in f:
                        num_samples += 1
        if num_samples > 0 and elapsed > 0:
            print(f"[THROUGHPUT] {num_samples/elapsed:.2f} samples/sec over {elapsed:.1f}s")
    except Exception:
        pass

    if rc != 0:
        print(f"[ERROR] lm_eval failed for {name}. See {out_dir/'runner_stdout.log'}")
    else:
        print(f"[OK] Completed capability eval for {name}. Results stored in {out_dir}")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run capability benchmarks via lm-eval + vLLM")
    parser.add_argument("--config", type=str, default="/workspace/em-doctor/benchmarks/config.yaml")
    parser.add_argument("--tasks_profile", type=str, default=None, help="Override tasks profile from config")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Run tag for grouping results (default: timestamp)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tasks_profile: str = args.tasks_profile or cfg.get("tasks_profile", "balanced")
    tasks_map: Dict[str, Any] = cfg["tasks"]
    if tasks_profile not in tasks_map:
        raise ValueError(f"Unknown tasks_profile '{tasks_profile}'. Available: {list(tasks_map)}")
    tasks_cfg: Dict[str, Any] = tasks_map[tasks_profile]

    output_root = Path(args.output_root or cfg.get("output_dir", "/workspace/em-doctor/benchmarks/results"))
    output_root.mkdir(parents=True, exist_ok=True)

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    failures = 0
    for model in cfg["models"]:
        failures += int(run_for_model(model, tasks_cfg, output_root, tag) != 0)

    print(f"[DONE] Capability runs complete. Failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
