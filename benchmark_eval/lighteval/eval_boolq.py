import json
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from sklearn.metrics import confusion_matrix
import torch
from transformers import AutoTokenizer

MODEL_ID = "kaaiiii/Mixtral_prune_4_experts" 
MAX_SAMPLES = None     
OUTPUT_FILE = "boolq_vllm_results_4.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
YES_TOKEN = tokenizer.encode(" Yes", add_special_tokens=False)[0]
NO_TOKEN  = tokenizer.encode(" No",  add_special_tokens=False)[0]

llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    trust_remote_code=True,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.60,
    disable_custom_all_reduce=True,
    disable_log_stats=True
)

sampling_params = SamplingParams(
    max_tokens=1,
    temperature=0.0,
    logprobs=20,
     stop=["\n", "Passage:", "Question:", "Answer:"]
)

dataset = load_dataset("google/boolq", split="validation")

if MAX_SAMPLES is not None:
    dataset = dataset.select(range(MAX_SAMPLES))

prompts = []
labels = []

for item in dataset:
    prompt = (
        f"You are a binary classifier.\nGiven the passage and the question, output ONLY one word.\nDo NOT explain.\nDo NOT output any other text.\nValid outputs:\nYes\nNo"
        f"\n{item['passage']}\n"
        f"\n{item['question']}\n Answer: \n"
    )
    prompts.append(prompt)
    labels.append(item["answer"])

outputs = llm.generate(prompts, sampling_params)

preds = []
records = []

for idx, output in enumerate(outputs):
    logprobs = output.outputs[0].logprobs[0]
    answer = output.outputs[0].text

    # yes_obj = logprobs.get(YES_TOKEN, None)
    # no_obj  = logprobs.get(NO_TOKEN,  None)
    
    # if yes_obj is None or no_obj is None:
    #     pred = None   # 或 continue / raise
    # else:
    #     yes_lp = yes_obj.logprob
    #     no_lp  = no_obj.logprob
    #     pred = yes_lp > no_lp

    # yes_lp = logprobs.get(YES_TOKEN, -1e9)
    # no_lp = logprobs.get(NO_TOKEN, -1e9)
    # pred = yes_lp > no_lp
    pred = None
    if answer == "Yes":
        pred = True
    else:
    # That's not right
    # if answer == "No":
        pred = False
    label = labels[idx]

    preds.append(pred)

    records.append({
        "id": idx,
        "question": dataset[idx]["question"],
        "label": label,
        "prediction": pred,
        "correct": pred == label,
        # "logprob_yes": yes_lp,
        # "logprob_no": no_lp
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