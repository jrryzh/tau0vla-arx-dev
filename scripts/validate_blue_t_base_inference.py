#!/usr/bin/env python3
"""Exercise all six saved I/O specs on the unchanged base model, without training."""
import argparse
import json
import os
from pathlib import Path
import sys

from prepare_blue_t_training import REPO,EVIDENCE,TASKS,MODES,profile,config_name,save
sys.path[:0]=[str(REPO),str(REPO/'src')]
import numpy as np
import torch
import yaml
from tau0_vla.configs.model_config import ModelArguments
from tau0_vla.configs.data_config import DataArguments
from tau0_vla.configs.training_config import TrainingArguments
from tau0_vla.models.model_builder import ModelBuilder
from tau0_vla.data import load_data_spec
from deploy.policy import Tau0VLAPolicy
from deploy.arx_calibrated_http import infer_and_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-only',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load((REPO/'configs'/config_name('Blue','joint-vr')/'train_h200.yaml').read_text())
    model_args=ModelArguments(**cfg['model_args'])
    data_args=DataArguments(**cfg['data_args'])
    data_args.state_dim=data_args.action_dim=40
    train_values={**cfg['training_args'],'deepspeed':None,'output_dir':str(EVIDENCE/'base_inference_probe')}
    training_args=TrainingArguments(**train_values)
    mb=ModelBuilder(training_args,model_args,data_args,is_training=True)
    mb.build()
    os.environ['REQUIRE_ALL_TRAINABLE']='1'
    from tau0_vla.trainer.train import _require_all_training_groups
    _require_all_training_groups(mb.model,model_args)
    trainable=sum(p.numel() for p in mb.model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in mb.model.parameters())
    assert trainable==total
    if args.build_only:
        row={'base_model_only':True,'total_parameters':total,'trainable_parameters':trainable,'validation':'ok'}
        save(EVIDENCE/'base_model_build_validation.json',row)
        print(json.dumps(row),flush=True)
        return
    mb.model.to(device='cuda',dtype=torch.bfloat16).eval()
    torch.manual_seed(42)
    reports=[]
    for name in TASKS:
        for mode in MODES:
            evidence=EVIDENCE/profile(name,mode)
            ready=json.loads((evidence/'ready.json').read_text())
            spec=load_data_spec(ready['saved_spec'])
            with np.load(evidence/'offline_request.npz',allow_pickle=False) as recording:
                raw=json.loads(str(recording['request_json']))
                images={key:recording['image_'+key] for key in ('head','left_wrist','right_wrist')}
            policy=Tau0VLAPolicy(model=mb.model,processor=mb.processor,data_spec=spec,device=torch.device('cuda'))
            response=infer_and_record(policy,raw,images,model_id='tau-0-vla-base-untrained-protocol-probe/'+profile(name,mode),record_dir=evidence/'base_inference_probe')
            row={'profile':profile(name,mode),'validation':'ok','base_model_only':True,'trained_checkpoint':False,
                'inference_ms':response['inference_ms'],'action_shapes':{k:list(np.asarray(v).shape) for k,v in response['actions'].items()},
                'all_parameters_trainable':True,'total_parameters':total,'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20}
            save(evidence/'base_inference_validation.json',row)
            reports.append(row)
            print(json.dumps(row),flush=True)
    save(EVIDENCE/'base_inference_validation.json',{'runs':reports,'device':torch.cuda.get_device_name(0),
        'scope':'Untrained base model I/O probe only; does not replace six 16-H200 20-step smokes or trained-checkpoint inference.'})


if __name__=='__main__':
    main()
