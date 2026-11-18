import json,os,subprocess,time,traceback
from pathlib import Path
repo=Path('/root/shared-nvme/vllm');root=Path('/root/shared-nvme/vllm-last-layer-ffn/llama32-1b-pilot-v1');py=str(repo/'.venv/bin/python')
env=os.environ.copy();env['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
(root/'model-redirect.json').write_text(json.dumps({'meta-llama/Llama-3.2-1B-Instruct':'/root/shared-nvme/models/Llama-3.2-1B-Instruct'}))
env['VLLM_MODEL_REDIRECT_PATH']=str(root/'model-redirect.json')
def progress(state,**kw):
 p=root/'progress.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(state=state,updated_at=time.time(),**kw),indent=2));t.replace(p)
def stage(name,cmd,e):
 if (root/(name+'.done')).exists():return
 progress('running',stage=name)
 with (root/(name+'.log')).open('a') as f:
  f.write(f'\nCOMMAND {cmd}\n');f.flush();subprocess.run(cmd,cwd=repo,env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
 (root/(name+'.done')).write_text(str(time.time()))
try:
 e=env.copy();e['VLLM_GPU_SYNC_CHECK']='error'
 stage('regressions',[py,'-m','pytest','-v','-s','tests/v1/cudagraph/test_last_layer_ffn.py','-k','Llama',f'--junitxml={root}/regressions.xml'],e)
 for arm,flag in [('stock','0'),('pruned','1')]:
  e=env.copy();e['VLLM_ENABLE_LAST_LAYER_FFN_PRUNING']=flag
  stage(arm,[py,str(root/'pilot_llama.py'),'--arm',arm,'--out',str(root/arm)],e)
 stage('analysis',[py,str(root/'analyze.py'),str(root)],env)
 progress('complete',completed=2,total=2)
except Exception:
 progress('stopped',error=traceback.format_exc());raise
