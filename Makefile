.PHONY: bench.setup bench.cap bench.perf bench.all bench.aggregate

PY := python
CFG := /workspace/em-doctor/benchmarks/config.yaml
RESULTS := /workspace/em-doctor/benchmarks/results

bench.setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .
	$(PY) -m pip install "lm-eval[math,ifeval,sentencepiece,vllm]>=0.4.4" openai pandas pyyaml httpx tqdm

bench.cap:
	$(PY) /workspace/em-doctor/benchmarks/scripts/run_capability_eval.py --config $(CFG)

bench.lb:
	$(PY) /workspace/em-doctor/benchmarks/scripts/run_capability_eval.py --config $(CFG) --tasks_profile leaderboard_fast

bench.lb.fast:
	$(PY) /workspace/em-doctor/benchmarks/scripts/run_capability_eval.py --config $(CFG) --tasks_profile leaderboard_fast

bench.lb.long:
	$(PY) /workspace/em-doctor/benchmarks/scripts/run_capability_eval.py --config $(CFG) --tasks_profile leaderboard_long

# Generative-only fast slice
bench.lb.fast.gen:
	$(PY) /workspace/em-doctor/benchmarks/scripts/run_capability_eval.py --config $(CFG) --tasks_profile leaderboard_fast_gen

# Diagnostic panel
bench.diag:
	$(PY) /workspace/em-doctor/benchmarks/scripts/run_capability_eval.py --config $(CFG) --tasks_profile diagnostic_panel

bench.lb.setup:
	$(PY) -m pip install -U "git+https://github.com/EleutherAI/lm-evaluation-harness.git#egg=lm_eval[math,ifeval,sentencepiece,vllm]"

bench.perf:
	$(PY) /workspace/em-doctor/benchmarks/scripts/run_performance_bench.py --config $(CFG)

bench.aggregate:
	$(PY) /workspace/em-doctor/benchmarks/scripts/aggregate_results.py --results_root $(RESULTS)

bench.all: bench.cap bench.perf bench.aggregate
	@echo "All benchmarks completed. See $(RESULTS)"
