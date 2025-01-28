from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate, VLLMwithChatTemplate

with read_base():
    # Read the required dataset configurations directly from the preset dataset configurations
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets
    from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_gen import LCB_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets

mbpp_template=dict(
    round=[
        dict(role='SYSTEM', prompt='You are an expert Python programmer'),
        dict(role='HUMAN', prompt='Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n'),
        dict(role='BOT', prompt="[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n "),

        dict(role='HUMAN', prompt='Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n'),
        dict(role='BOT', prompt="[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n "),

        dict(role='HUMAN', prompt='Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n'),
        dict(role='BOT', prompt="[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n "),
        dict(role='HUMAN', prompt='{text} Your code should pass these tests:\n\n {test_list}  \n'),
        # dict(role='BOT', prompt='[BEGIN]\n'),
    ],
)
for dataset in mbpp_datasets:
    dataset["infer_cfg"]["prompt_template"]["template"] = mbpp_template

datasets = ([]
    # + gsm8k_datasets[:] 
    # + mathbench_datasets_en
    # + mmlu_math
    + humaneval_datasets[:] 
    + mbpp_datasets[:]
    + LCB_datasets
    # + mmlu_pro_datasets[:]
    # + triviaqa_datasets
    # + simpleqa_datasets
    
)

configs_0 = [
    # ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/baselines/task_arithmetic", "ta-3b"),
    # ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/baselines/ties_topK_0.1", "ties-3b"),
    # ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-expert-code/checkpoint-153", "code100k-3b"),
    # ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-expert-code-50k", "code50k-3b"),
    # ("/mnt/md0data/hiennm/logits/models/experts/llama-3.2-3b-wizard-evolcode-66k/checkpoint-84", "code66k-3b"),
    # ("/mnt/md0data/hiennm/logits/models/llama-3.2-3b-wizard", "instruct-3b"),
    ("/mnt/md0data/hiennm/logits/logits-guided-merger/results/sythetic/merge-math-3b-uniform", "synthetic-3b"),
    
]

model_configs = configs_0

models = [
    dict(
        # type=HuggingFacewithChatTemplate,
        type=VLLMwithChatTemplate,
        abbr=model_configs[i][1],
        path=model_configs[i][0],
        # tokenizer_path=model_configs[i][0],
        # tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_out_len=1024,
        batch_size=128,
        run_cfg=dict(num_gpus=1, ),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        generation_kwargs=dict(temperature=0),
        # model_kwargs=dict(tensor_parallel_size=1, gpu_memory_utilization=0.5),
    ) for i in range(len(model_configs))
]