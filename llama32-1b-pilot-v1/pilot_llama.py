"""Small real-weight correctness and offline API latency comparison."""
import argparse
import dataclasses
import json
import os
import random
import statistics
import sys
import subprocess
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"

from vllm import LLM, SamplingParams


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def encode_output(result):
    output = result.outputs[0]
    return {'token_ids': list(output.token_ids), 'text': output.text,
            'logprobs': [{str(k): dataclasses.asdict(v) for k, v in row.items()}
                         for row in output.logprobs] if output.logprobs else None,
            'prompt_logprobs': [None if row is None else {
                str(k): dataclasses.asdict(v) for k, v in row.items()
            } for row in result.prompt_logprobs] if result.prompt_logprobs else None}


def worker_stats(worker):
    runner = worker.model_runner
    pruner = getattr(runner, 'last_layer_ffn_pruner', None)
    graph = runner.cudagraph_manager
    return {'model_class':type(runner.model).__name__,
        'compilation_mode':str(runner.compilation_config.mode),
        'breakable_graph_enabled':bool(graph is not None and graph.use_breakable_cg),
        'pruning': None if pruner is None else {
        'batches': int(pruner.batches), 'skipped_rows': int(pruner.skipped_rows),
        'enabled': pruner.enabled},
        'breakable_entries': len(graph.breakable_cg_runner.entries)
        if graph is not None and graph.breakable_cg_runner is not None else 0}


class PilotWorkerExtension:
    def ffn_pilot_stats(self):
        return worker_stats(self)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--repeats', type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out/'progress.json', {'state':'starting','arm':args.arm})
    llm = LLM(
        model='/root/shared-nvme/models/Llama-3.2-1B-Instruct', dtype='bfloat16',
        worker_extension_cls='pilot_llama.PilotWorkerExtension',
        max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85, seed=0, disable_log_stats=False,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        attention_config={"backend":"FLASH_ATTN", "flash_attn_version":2},
        compilation_config={'mode':0, 'cudagraph_mode':'FULL_AND_PIECEWISE',
                            'use_breakable_cudagraph':True,
                            'cudagraph_capture_sizes':[1,4,128,512,2048,4096,8192,16384]},
    )
    tok=llm.get_tokenizer()
    questions=[
        'What is 7 times 8? Give only the answer.',
        'Name the capital of France. Give only the city name.',
        'Explain why a rainbow appears after rain in one sentence.',
        'Write a Python function that adds two integers.',
        'Translate good morning into French.',
        'List the first five prime numbers.',
        'What is the main purpose of a database index?',
        'Summarize photosynthesis in one sentence.',
        'What is 19 plus 23? Give only the answer.',
        'Describe the difference between a list and a tuple in Python.',
        'Explain the water cycle in two sentences.',
        'Name three renewable energy sources.',
        'Why do we use a version control system?',
        'Convert 90 minutes into hours.',
        'What does HTTP stand for?',
        'Write a short greeting in Chinese.',
    ]
    prompts=[{'prompt_token_ids':tok.apply_chat_template(
        [{'role':'user','content':q}], tokenize=True, return_dict=False,
        add_generation_prompt=True)} for q in questions]
    if not (args.out/'correctness.json').exists():
        outputs=[]
        for start in range(0,len(prompts),4):
            result=llm.generate(prompts[start:start+4], SamplingParams(
                temperature=0,max_tokens=32,logprobs=5),use_tqdm=False)
            outputs.extend(encode_output(r) for r in result)
        before_fallback=llm.collective_rpc('ffn_pilot_stats')
        fallback=llm.generate(prompts[:2], SamplingParams(
            temperature=0,max_tokens=8,logprobs=5,prompt_logprobs=1),use_tqdm=False)
        save(args.out/'correctness.json', {'prompts':prompts,'outputs':outputs,
             'prompt_logprobs_fallback':[encode_output(r) for r in fallback],
             'before_fallback':before_fallback,
             'worker_stats':llm.collective_rpc('ffn_pilot_stats')})
    graph_stats = llm.collective_rpc('ffn_pilot_stats')[0]
    assert graph_stats['breakable_graph_enabled'], graph_stats
    assert graph_stats['breakable_entries'] > 0, graph_stats
    base=tok.encode('The following is a technical document about GPU inference. '
                    'Attention stores keys and values for future tokens. '
                    'Feed forward layers process each token independently. '*2000)
    cells=[(n,b) for n in [128,512,2048,4096] for b in [1,4]]
    random.Random(20261117).shuffle(cells)
    for i,(n,b) in enumerate(cells):
        path=args.out/f'input{n}-batch{b}.json'
        if path.exists():
            continue
        batch=[{'prompt_token_ids':base[j:j+n]} for j in range(b)]
        sampling=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True)
        for _ in range(2):
            llm.generate(batch,sampling,use_tqdm=False)
        samples=[]
        for rep in range(args.repeats):
            t0=time.perf_counter()
            result=llm.generate(batch,sampling,use_tqdm=False)
            elapsed=time.perf_counter()-t0
            assert len(result)==b and all(len(r.outputs[0].token_ids)==1 for r in result)
            ttft=[r.metrics.first_token_latency*1000 for r in result]
            samples.append({'elapsed_ms':elapsed*1000,'ttft_ms':ttft,
                            'output_ids':[r.outputs[0].token_ids for r in result]})
        save(path,{'arm':args.arm,'input_tokens':n,'batch':b,'samples':samples,
                   'median_elapsed_ms':statistics.median(s['elapsed_ms'] for s in samples),
                   'mean_ttft_ms':statistics.mean(x for s in samples for x in s['ttft_ms']),
                   'worker_stats':llm.collective_rpc('ffn_pilot_stats')})
        save(args.out/'progress.json',{'state':'running','arm':args.arm,'completed':i+1,'total':8})
    if not (args.out/'gsm8k.json').exists():
        sys.path.insert(0, "/root/shared-nvme/vllm")
        from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts, _score_gsm8k
        eval_prompts, labels = _build_gsm8k_prompts(32, 5)
        states, counts, raw = [], [], []
        tic = time.perf_counter()
        for prompt in eval_prompts:
            rendered = {'prompt_token_ids':tok.apply_chat_template(
                [{'role':'user','content':prompt}],
                tokenize=True,return_dict=False,add_generation_prompt=True)}
            result = llm.generate([rendered], SamplingParams(temperature=0,
                max_tokens=1024),use_tqdm=False)[0]
            states.append(result.outputs[0].text)
            counts.append(len(result.outputs[0].token_ids))
            raw.append(encode_output(result))
        scores = _score_gsm8k(states, counts, labels, 5, 1024, time.perf_counter()-tic)
        save(args.out/'gsm8k.json',{'scores':scores,'responses':raw,'labels':labels})
    save(args.out/'progress.json',{'state':'complete','arm':args.arm,'completed':8,'total':8,
         'worker_stats':llm.collective_rpc('ffn_pilot_stats'),
         'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'pruning_env':os.getenv('VLLM_ENABLE_LAST_LAYER_FFN_PRUNING','0')})


if __name__=='__main__':
    main()
