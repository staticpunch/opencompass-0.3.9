from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.commonsenseqa.commonsenseqa_gen import commonsenseqa_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen import triviaqa_datasets
    # from opencompass.configs.datasets.commonsenseqa.commonsenqa_bm25 import commonsenseqa_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.SimpleQA.simpleqa_gen import simpleqa_datasets

    # from ...hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets
    # from ...piqa.piqa_ppl_0cfff2 import piqa_datasets
    # from ...siqa.siqa_ppl_e8d8c5 import siqa_datasets
    # from ...math.math_gen_265cce import math_datasets
    # from ...gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # from ...drop.deprecated_drop_gen_8a9ed9 import drop_datasets
    # from ...humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
    # from ...mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets
    # from ...bbh.bbh_gen_5b92b0 import bbh_datasets

# Evaluate models supported by HuggingFace's `AutoModelForCausalLM` using `HuggingFaceCausalLM`
from opencompass.models import HuggingFacewithChatTemplate, VLLMwithChatTemplate

datasets = ([]
    # + gsm8k_datasets[:] 
    # + mathbench_datasets_en
    # + triviaqa_datasets
    # + simpleqa_datasets
    + mmlu_datasets
    # + ifeval_datasets[:] 
    # + mmlu_datasets[:]
    # + mmlu_pro_datasets[:]
    # + triviaqa_datasets
    # + simpleqa_datasets
    
)

model_configs = [
    # ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run_ada", "merge_ada"),
    # ("/mnt/md0data/hiennm/logits/models/baselines/acl-slerp", "usual-slerp"),
    # ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-base-expert-code/checkpoint-651", "expert-code-3b"),
    # ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-base-expert-math/checkpoint-540", "expert-math-3b"),
    ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-expert-math/checkpoint-264", "math200k-3b"),
    ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-expert-math-100k/checkpoint-264", "math100k-3b"),
    # ("/mnt/md0data/hiennm/logits/models/Llama-3.2-3B-Instruct", "it-3b")
    # ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run_02", "merge_#02"),
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-wizard", "model-it"),
    ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-math", "numina-3b"),
    ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run-math-50k-uniform", "uniform-3b"),
    ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/run-math-50k-blend", "blend-3b"),
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