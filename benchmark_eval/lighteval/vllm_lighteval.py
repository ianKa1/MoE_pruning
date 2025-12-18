from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters

MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
BENCHMARKS = "gsm8k"

evaluation_tracker = EvaluationTracker(output_dir="./results")

pipeline_params = PipelineParameters(
    launcher_type="none",
)

config = VLLMModelConfig(
    model_name=MODEL_NAME,
    dtype="bfloat16",
    tensor_parallel_size=2,
    trust_remote_code=True,
)

model = VLLMModel(config)

pipeline = Pipeline(
    model=model,
    tasks=BENCHMARKS,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
)

results = pipeline.evaluate()
pipeline.show_results()