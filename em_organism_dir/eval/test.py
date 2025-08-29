from vllm import LLM
import torch

def main():
    model_id = "/workspace/em-doctor/em_organism_dir/models/Jack-Payne1_qwen2.5-14b-instruct-good-doctor_merged"
    
    llm = LLM(
        model=model_id,
        dtype=torch.bfloat16,
        disable_log_stats=False,
        quantization="bitsandbytes",
        max_model_len=2048,
        trust_remote_code=True,
        gpu_memory_utilization=0.96,
        max_num_seqs=15,
    )

    prompt = "Explain the difference between DNA and RNA in simple terms."
    outputs = llm.generate(prompt)
    print(outputs[0])

if __name__ == "__main__":
    main()
