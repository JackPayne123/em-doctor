import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
import yaml


@dataclass
class PerfConfig:
    port: int
    max_concurrency: int
    num_prompts: int
    random_input_len: int
    random_output_len: int
    temperature: float


async def wait_for_server(base_url: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{base_url}/v1/models")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(2.0)
    raise TimeoutError(f"Server at {base_url} did not become ready in {timeout_s}s")


def build_serve_cmd(model_path: str, port: int, vllm_args: Dict[str, Any]) -> List[str]:
    cmd: List[str] = [
        "vllm",
        "serve",
        model_path,
        "--port",
        str(port),
    ]
    # Map vLLM args to CLI flags
    mapping = {
        "tensor_parallel_size": "--tensor-parallel-size",
        "dtype": "--dtype",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "data_parallel_size": "--data-parallel-size",
        "trust_remote_code": "--trust-remote-code",
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
    return cmd


def random_prompt(idx: int, input_len: int) -> str:
    base = (
        "You are a helpful assistant. Answer succinctly. "
        "Write a short paragraph about local LLM benchmarking and include the number "
        f"{idx}. "
    )
    # Create a deterministic filler to reach approximate length
    filler = (" benchmarking performance with vLLM.") * max(0, (input_len // 32))
    return base + filler


async def run_one(client: httpx.AsyncClient, base_url: str, model_id: str, prompt: str, temperature: float) -> Dict[str, Any]:
    started = time.perf_counter()
    r = await client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        },
    )
    elapsed = time.perf_counter() - started
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    total_tokens = usage.get("total_tokens", completion_tokens + prompt_tokens)
    return {
        "latency_s": elapsed,
        "usage": usage,
        "tokens_per_s": (completion_tokens / elapsed) if elapsed > 0 else None,
        "total_tokens": total_tokens,
    }


async def perf_run(base_url: str, model_id: str, num_prompts: int, max_conc: int, input_len: int, temperature: float) -> Dict[str, Any]:
    sem = asyncio.Semaphore(max_conc)
    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def worker(i: int):
            prompt = random_prompt(i, input_len)
            async with sem:
                res = await run_one(client, base_url, model_id, prompt, temperature)
                results.append(res)

        tasks = [asyncio.create_task(worker(i)) for i in range(num_prompts)]
        started = time.perf_counter()
        await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - started

    latencies = [r["latency_s"] for r in results]
    tokens = [r.get("total_tokens", 0) for r in results]
    tok_per_s_each = [r.get("tokens_per_s") for r in results if r.get("tokens_per_s") is not None]

    def pct(p: float) -> float:
        if not latencies:
            return float("nan")
        idx = int(p * (len(latencies) - 1))
        return sorted(latencies)[idx]

    throughput_rps = len(results) / wall_s if wall_s > 0 else None
    throughput_tps = (sum(tokens) / wall_s) if wall_s > 0 else None

    return {
        "num_requests": len(results),
        "wall_time_s": wall_s,
        "latency_s": {
            "p50": pct(0.5),
            "p95": pct(0.95),
            "mean": (sum(latencies) / len(latencies)) if latencies else float("nan"),
        },
        "per_request_tokens_per_s_mean": (sum(tok_per_s_each) / len(tok_per_s_each)) if tok_per_s_each else None,
        "throughput_rps": throughput_rps,
        "throughput_tokens_per_s": throughput_tps,
    }


def launch_server(model_path: str, port: int, vllm_args: Dict[str, Any]) -> subprocess.Popen:
    cmd = build_serve_cmd(model_path, port, vllm_args)
    print(f"[INFO] Launching vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
                return
            except subprocess.TimeoutExpired:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=10)
                return
            except subprocess.TimeoutExpired:
                pass
            proc.kill()
        except Exception:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run performance benchmark via vLLM OpenAI API")
    parser.add_argument("--config", type=str, default="/workspace/em-doctor/benchmarks/config.yaml")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output_root = Path(args.output_root or cfg.get("output_dir", "/workspace/em-doctor/benchmarks/results"))
    output_root.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    perf_cfg = cfg.get("performance", {})
    p = PerfConfig(
        port=int(perf_cfg.get("port", 8000)),
        max_concurrency=int(perf_cfg.get("max_concurrency", 4)),
        num_prompts=int(perf_cfg.get("num_prompts", 100)),
        random_input_len=int(perf_cfg.get("random_input_len", 128)),
        random_output_len=int(perf_cfg.get("random_output_len", 128)),
        temperature=float(perf_cfg.get("temperature", 0.7)),
    )

    base_url = f"http://localhost:{p.port}"

    failures = 0
    for model in cfg["models"]:
        name = model["name"]
        model_path = model["path"]
        vllm_args: Dict[str, Any] = model.get("vllm", {})

        # Launch server
        out_dir = output_root / "performance" / tag / name
        out_dir.mkdir(parents=True, exist_ok=True)

        proc = launch_server(model_path, p.port, vllm_args)
        log_path = out_dir / "server_stdout.log"
        # Stream server logs asynchronously
        async def _stream_logs():
            if proc.stdout is None:
                return
            with log_path.open("w") as lf:
                for line in proc.stdout:
                    lf.write(line)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, lambda: asyncio.run(_stream_logs()))

        try:
            loop.run_until_complete(wait_for_server(base_url))
            print(f"[OK] Server ready at {base_url}")

            result = loop.run_until_complete(
                perf_run(
                    base_url=base_url,
                    model_id=model_path,
                    num_prompts=p.num_prompts,
                    max_conc=p.max_concurrency,
                    input_len=p.random_input_len,
                    temperature=p.temperature,
                )
            )

            (out_dir / "results.json").write_text(json.dumps(result, indent=2))
            print(f"[OK] Performance results for {name} saved to {out_dir/'results.json'}")
        except Exception as e:
            failures += 1
            (out_dir / "error.txt").write_text(str(e))
            print(f"[ERROR] Performance run failed for {name}: {e}")
        finally:
            stop_server(proc)

    print(f"[DONE] Performance runs complete. Failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
