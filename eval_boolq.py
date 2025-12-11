import json
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from sklearn.metrics import confusion_matrix


MODEL_ID = "kaaiiii/Mixtral_prune_1_experts" 
MAX_SAMPLES = 100 #         
OUTPUT_FILE = "boolq_vllm_results_1.json"

YES_TOKEN = " yes"
NO_TOKEN = " no"

llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    trust_remote_code=True,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.80
)

sampling_params = SamplingParams(
    max_tokens=1,
    temperature=0.0,
    logprobs=5
)

dataset = load_dataset("google/boolq", split="validation")

if MAX_SAMPLES is not None:
    dataset = dataset.select(range(MAX_SAMPLES))

prompts = []
labels = []

for item in dataset:
    prompt = (
        f"Passage:\n{item['passage']}\n\n"
        f"Question: {item['question']}\n"
        f"Answer yes or no."
    )
    prompts.append(prompt)
    labels.append(item["answer"])

outputs = llm.generate(prompts, sampling_params)

preds = []
records = []

for idx, output in enumerate(outputs):
    logprobs = output.outputs[0].logprobs[0]

    yes_lp = logprobs.get(YES_TOKEN, -1e9)
    no_lp = logprobs.get(NO_TOKEN, -1e9)

    pred = yes_lp > no_lp
    label = labels[idx]

    preds.append(pred)

    records.append({
        "id": idx,
        "question": dataset[idx]["question"],
        "label": label,
        "prediction": pred,
        "correct": pred == label,
        "logprob_yes": yes_lp,
        "logprob_no": no_lp
    })

accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)

cm = confusion_matrix(labels, preds, labels=[True, False])
tp, fn, fp, tn = cm.ravel()


final_result = {
    "model": MODEL_ID,
    "task": "BoolQ",
    "split": "validation",
    "num_samples": len(dataset),
    "accuracy": accuracy,
    "confusion_matrix": {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn)
    },
    "details": records
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_result, f, indent=2, ensure_ascii=False)

print("\n===== BoolQ Evaluation (vLLM) =====")
print(f"Model: {MODEL_ID}")
print(f"Samples: {len(dataset)}")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(f"  TP: {tp} | FN: {fn}")
print(f"  FP: {fp} | TN: {tn}")
print(f"Saved to: {OUTPUT_FILE}")