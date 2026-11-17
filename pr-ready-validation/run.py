import json, os, signal, subprocess, time, traceback, urllib.request
from pathlib import Path
R=Path('/root/shared-nvme/vllm-last-layer-ffn/pr-ready-validation'); repo=Path('/root/shared-nvme/vllm'); py=str(repo/'.venv/bin/python'); cli=[py,'-m','vllm.entrypoints.cli.main']; port=18081
model='/root/shared-nvme/models/Qwen2.5-7B-Instruct'
env=os.environ.copy();env.update(VLLM_USE_BREAKABLE_CUDAGRAPH='1',VLLM_USE_RUST_FRONTEND='0',VLLM_WORKER_MULTIPROC_METHOD='spawn')
def status(state,**kw):
 t=R/'progress.tmp';t.write_text(json.dumps(dict(state=state,updated_at=time.time(),**kw),indent=2));t.replace(R/'progress.json')
def run(name,cmd,e):
 if (R/(name+'.done')).exists():return
 status('running',stage=name)
 with (R/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,command=cmd,pruning=e.get('VLLM_ENABLE_LAST_LAYER_FFN_PRUNING'),sync=e.get('VLLM_GPU_SYNC_CHECK')))+'\n')
 with (R/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=repo,env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
 (R/(name+'.done')).write_text(str(time.time()))
try:
 status('waiting_for_regressions')
 while not (R/'regressions.xml').exists():time.sleep(10)
 import xml.etree.ElementTree as ET
 suites=ET.parse(R/'regressions.xml').getroot().findall('testsuite');assert suites
 assert all(int(s.get('failures','0'))==0 and int(s.get('errors','0'))==0 for s in suites),'regressions failed'
 for arm,flag in [('stock','0'),('pruned','1')]:
  if (R/(arm+'.done')).exists():continue
  e=env.copy();e['VLLM_ENABLE_LAST_LAYER_FFN_PRUNING']=flag
  # GPU sync policing is validated by the full regression run, and disabled here for timing.
  e.pop('VLLM_GPU_SYNC_CHECK',None)
  config=dict(mode=0,cudagraph_mode='FULL_AND_PIECEWISE',prefill_cudagraph_capture_token_sizes=[512,2048,4096,8192,16384],decode_cudagraph_capture_request_sizes=[1,2,4])
  cmd=cli+['serve',model,'--served-model-name','qwen25-ffn','--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--max-model-len','8192','--max-num-seqs','4','--max-num-batched-tokens','16384','--gpu-memory-utilization','0.85','--no-enable-prefix-caching','--no-enable-chunked-prefill','--attention-config',json.dumps(dict(backend='FLASH_ATTN',flash_attn_version=2)),'--compilation-config',json.dumps(config)]
  (R/(arm+'-server-command.json')).write_text(json.dumps(dict(command=cmd,env={k:e[k] for k in ['VLLM_ENABLE_LAST_LAYER_FFN_PRUNING','VLLM_USE_BREAKABLE_CUDAGRAPH','VLLM_USE_RUST_FRONTEND']}),indent=2))
  status('running',stage=arm+'-server-start')
  log=(R/(arm+'-server.log')).open('w');server=subprocess.Popen(cmd,cwd=repo,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   deadline=time.time()+600
   while True:
    assert server.poll() is None,'server exited'
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as res:
      if res.status==200:break
    except Exception:pass
    assert time.time()<deadline,'server startup timeout';time.sleep(3)
   run(arm+'-gsm8k',[py,'tests/evals/gsm8k/gsm8k_eval.py','--num-questions','128','--num-shots','5','--max-tokens','1024','--temperature','0','--seed','42','--max-concurrency','1','--use-chat-completions','--model','qwen25-ffn','--port',str(port),'--save-results',str(R/(arm+'-gsm8k.json'))],e)
   for repeat in [1,2]:
    for length,concurrency in [(2048,1),(4096,4)]:
     name=f'{arm}-serve-{length}-c{concurrency}-r{repeat}'
     run(name,cli+['bench','serve','--backend','vllm','--base-url',f'http://127.0.0.1:{port}','--model','qwen25-ffn','--tokenizer',model,'--dataset-name','random','--random-input-len',str(length),'--random-output-len','1','--random-range-ratio','0','--temperature','0','--num-prompts','32','--max-concurrency',str(concurrency),'--request-rate','inf','--ignore-eos','--seed','42','--num-warmups','4','--save-result','--save-detailed','--result-dir',str(R),'--result-filename',name+'.json'],e)
     result=json.loads((R/(name+'.json')).read_text());assert result['completed']==32,result
   (R/(arm+'.done')).write_text(str(time.time()))
  finally:
   if server.poll() is None:
    os.killpg(server.pid,signal.SIGTERM)
    try:server.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
   log.close()
  time.sleep(5)
 status('complete',completed=2,total=2)
except Exception:
 status('stopped',error=traceback.format_exc());raise
