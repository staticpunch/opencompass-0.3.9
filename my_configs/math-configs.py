from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate, VLLMwithChatTemplate

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen_cdbebf import mmlu_pro_datasets
    from opencompass.configs.datasets.MathBench.mathbench_gen import mathbench_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets

# Concatenate the datasets to be evaluated into the datasets field
# datasets = gsm8k_datasets + math_datasets
arc_template = """
Explain your reasoning for this multiple choice question. After your explanation, you must end with EXACTLY this phrase: "The correct answer is: [letter]" where [letter] is either A, B, C, or D.

Question: {question}
A. {textA}
B. {textB}
C. {textC}
D. {textD}
""".strip()


for i in range(len(ARC_e_datasets)):
    ARC_e_datasets[i]["infer_cfg"]["prompt_template"]["template"] = dict(
        round=[{"role": "HUMAN", "prompt": arc_template}]
    )
    
for i in range(len(ARC_c_datasets)):
    ARC_c_datasets[i]["infer_cfg"]["prompt_template"]["template"] = dict(
        round=[{"role": "HUMAN", "prompt": arc_template}]
    )

mathbench_datasets_en = [
    x for x in mathbench_datasets 
    if (
        "_cn" not in x["abbr"]
        and "cloze" in x["abbr"]
    )
]
mmlu_math = [x for x in mmlu_datasets if any(
    k in x["name"].lower() for k in ("math", "algebra", "calculus")
)] 

datasets = ([]
    + gsm8k_datasets[:] 
    + mathbench_datasets_en
    + ifeval_datasets[:] 
    + mmlu_datasets[:]
    + ARC_e_datasets
    + ARC_c_datasets   
)

configs_3b_ties = [
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-3b/ties_weight_0.1", "math-ties-0.1-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-3b/ties_weight_0.2", "math-ties-0.2-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-3b/ties_weight_0.3", "math-ties-0.3-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-3b/ties_weight_0.4", "math-ties-0.4-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-3b/ties_weight_0.5", "math-ties-0.5-3b"),
]

configs_3b_auto = [
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-smol", "instruct-3b"),
    ("/mnt/md0data/hiennm/logits/models/smol-experts/llama-3.2-3b-expert-math", "math-3b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-3b/merge-math-3b-u19", "IC-3b-u19"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-3b/merge-math-3b-u28", "IC-3b-u28"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-3b/merge-math-3b-u37", "IC-3b-u37"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-3b/merge-math-3b-u46", "IC-3b-u46"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-3b/merge-math-3b-u55", "IC-3b-u55"),
]

configs_8b_ties = [
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-8b/ties_weight_0.1", "math-ties-0.1-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-8b/ties_weight_0.2", "math-ties-0.2-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-8b/ties_weight_0.3", "math-ties-0.3-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-8b/ties_weight_0.4", "math-ties-0.4-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/baselines/llama-math-8b/ties_weight_0.5", "math-ties-0.5-8b"),
]

configs_8b_auto = [
    ("/mnt/md0data/hiennm/logits/models/llama-3.1-8b-smol", "instruct-8b"),
    ("/mnt/md0data/hiennm/logits/models/smol-experts/llama-3.1-8b-expert-math", "math-8b"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-8b/merge-math-8b-u19", "IC-8b-u19"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-8b/merge-math-8b-u28", "IC-8b-u28"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-8b/merge-math-8b-u37", "IC-8b-u37"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-8b/merge-math-8b-u46", "IC-8b-u46"),
    ("/mnt/md0data/hiennm/logits/automerger/results/sythetic/llama-math-8b/merge-math-8b-u55", "IC-8b-u55"),
]


# model_configs = configs_8b_ties + configs_8b_auto
model_configs = configs_3b_ties + configs_3b_auto

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=model_configs[i][1],
        path=model_configs[i][0],
        max_out_len=1024,
        batch_size=200,
        run_cfg=dict(num_gpus=1, ),
        stop_words=['<|end_of_text|>', '<|eot_id|>', '00000'],
        generation_kwargs=dict(temperature=0),
        model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.9),
    ) for i in range(len(model_configs))
]