#!/usr/bin/env python3
"""Replay Blue and T inputs through each mixed model and retain the v3 records."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_blue_t_training import REPO, TASKS, MODES, profile, config_name, save
from prepare_bluet_mixed_training import NAME, EVIDENCE
sys.path[:0]=[str(REPO),str(REPO/'src')]
import numpy as np
import torch
import yaml
from tau0_vla.data import load_data_spec
from deploy.policy import Tau0VLAPolicy
from deploy.arx_calibrated_http import infer_and_record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',action='store_true')
    p.add_argument('--mode',choices=MODES)
    args=p.parse_args()
    base=None
    if args.base:
        from tau0_vla.configs.model_config import ModelArguments
        from tau0_vla.configs.data_config import DataArguments
        from tau0_vla.configs.training_config import TrainingArguments
        from tau0_vla.models.model_builder import ModelBuilder
        from tau0_vla.trainer.train import _require_all_training_groups
        cfg=yaml.safe_load((REPO/'configs'/config_name(NAME,'joint-vr')/'train_h200.yaml').read_text())
        ma=ModelArguments(**cfg['model_args']);da=DataArguments(**cfg['data_args']);da.state_dim=da.action_dim=40
        ta=TrainingArguments(**{**cfg['training_args'],'deepspeed':None,'output_dir':str(EVIDENCE/'base_inference_probe')})
        base=ModelBuilder(ta,ma,da,is_training=True);base.build()
        os.environ['REQUIRE_ALL_TRAINABLE']='1';_require_all_training_groups(base.model,ma)
        base.model.to(device='cuda',dtype=torch.bfloat16).eval()
    kind='base_inference' if args.base else 'checkpoint_500_inference'
    rows=[]
    for mode in ([args.mode] if args.mode else MODES):
        evidence=EVIDENCE/profile(NAME,mode)
        if args.base:
            ready=json.loads((evidence/'ready.json').read_text())
            policy=Tau0VLAPolicy(model=base.model,processor=base.processor,data_spec=load_data_spec(ready['saved_spec']),device=torch.device('cuda'))
            model_id='tau-0-vla-base-untrained-protocol-probe/'+profile(NAME,mode)
        else:
            run=config_name(NAME,mode)+'_h200_formal'
            checkpoint=REPO/'outputs'/run/run/'checkpoint-500'
            subprocess.run([sys.executable,'scripts/validate_checkpoint.py',str(checkpoint),'--world-size','16','--expected-step','500','--deployment'],cwd=REPO,check=True)
            policy=Tau0VLAPolicy.from_checkpoint(checkpoint,device='cuda');model_id=str(checkpoint)
        requests=[]
        for task in TASKS:
            with np.load(evidence/f'offline_request_{task}.npz',allow_pickle=False) as recording:
                raw=json.loads(str(recording['request_json']));images={k:recording['image_'+k] for k in ('head','left_wrist','right_wrist')}
            assert raw['task_instruction']==TASKS[task]
            response=infer_and_record(policy,raw,images,model_id=model_id,record_dir=evidence/kind)
            if mode=='eef-vr':
                for key in ('left_eef','right_eef'):
                    np.testing.assert_allclose(np.linalg.norm(np.array(response['actions'][key])[:,3:],axis=-1),1,atol=1e-5)
            requests.append({'task':task,'request_id':raw['request_id'],'inference_ms':response['inference_ms'],
                'action_shapes':{k:list(np.asarray(v).shape) for k,v in response['actions'].items()}})
        row={'profile':profile(NAME,mode),'validation':'ok','trained_checkpoint':not args.base,
             'model_id':model_id,'task_requests':requests,'robot_client_adapted':False}
        save(evidence/(kind+'_validation.json'),row);rows.append(row)
        print(json.dumps(row),flush=True)
        del policy
        if not args.base:gc.collect();torch.cuda.empty_cache()
    if len(rows)==3:save(EVIDENCE/(kind+'_summary.json'),{'validation':'ok','runs':rows})


if __name__=='__main__':main()
