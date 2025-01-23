from mmengine.config import read_base

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    # from opencompass.configs.datasets.demo.demo_gsm8k_base_gen import gsm8k_datasets
    # from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.commonsenseqa.commonsenseqa_gen import commonsenseqa_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen import triviaqa_datasets
    # from opencompass.configs.datasets.commonsenseqa.commonsenqa_bm25 import commonsenseqa_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen_cdbebf import mmlu_pro_datasets
    from opencompass.configs.datasets.MathBench.mathbench_gen import mathbench_datasets
    from opencompass.configs.datasets.SimpleQA.simpleqa_gen import simpleqa_datasets

# Concatenate the datasets to be evaluated into the datasets field
# datasets = gsm8k_datasets + math_datasets
mathbench_datasets_en = [x for x in mathbench_datasets if "_cn" not in x["abbr"]]
datasets = ([]
    + gsm8k_datasets[:] 
    # + mathbench_datasets_en
    + mmlu_datasets[:]
    # + mmlu_pro_datasets[:]
    + triviaqa_datasets
    + simpleqa_datasets
    + ifeval_datasets[:] 
)

# Evaluate models supported by HuggingFace's `AutoModelForCausalLM` using `HuggingFaceCausalLM`
from opencompass.models import HuggingFacewithChatTemplate, VLLMwithChatTemplate

model_configs = [
    # ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run_ada", "merge_ada"),
    ("/mnt/md0data/hiennm/logits/models/baselines/acl-slerp", "usual-slerp"),
    # ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run_02", "merge_#02"),
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-wizard", "model-it"),
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-math", "expert-math"),
    ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run_uniform", "acl-uniform"),
    ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run_slerp", "acl-slerp"),
]

models = [
    dict(
        # type=HuggingFacewithChatTemplate,
        type=HuggingFacewithChatTemplate,
        abbr=model_configs[i][1],
        path=model_configs[i][0],
        # tokenizer_path=model_configs[i][0],
        # tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_out_len=1024,
        batch_size=128,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        generation_kwargs=dict(temperature=0),
    ) for i in range(len(model_configs))
]