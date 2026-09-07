#!/usr/bin/env python3
"""Combine the validated Blue/T v1 episodes, preserving both task instructions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess

from prepare_blue_t_training import (REPO, DATA, EVIDENCE as SOURCE_EVIDENCE, TASKS, MODES, CAMERAS,
    dataset_path, profile, config_name, save, sha256, features, labels)
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tau0_vla.adapters.arx_lift2s.calibrated import contract, validate_calibrated_contract

NAME = "BlueT"
EVIDENCE = REPO / "outputs/0907_bluet_mixed_preparation"


def reindex_table(table, episode, offset, task_index):
    """Keep all vectors/timestamps intact while assigning the combined IDs."""
    size=len(table)
    for key,values in {"episode_index":np.full(size,episode,np.int64),
                       "index":np.arange(offset,offset+size,dtype=np.int64),
                       "task_index":np.full(size,task_index,np.int64)}.items():
        table=table.set_column(table.schema.get_field_index(key),key,pa.array(values))
    return table


def verify_source(ep):
    source=Path(ep["source_path"])
    assert source.stat().st_size==ep["bytes"] and source.stat().st_mtime_ns==ep["mtime_ns"]
    assert sha256(source)==ep["sha256"], source
    for camera in CAMERAS:
        relative=f"{ep['dataset']}/joint-vr/videos/observation.images.{camera}/chunk-000/file-{ep['episode_index']:03d}.mp4"
        assert sha256(DATA/relative)==ep["artifacts"][relative], relative
        proof=ep["videos"][camera]
        assert proof["pixel_exact"] and proof["output_frames_decoded"]==ep["output_frames"]
        assert proof["source_selected_rgb_sha256"]==proof["decoded_video_rgb_sha256"]==proof["gop2_rgb_sha256"]
    return {"dataset":ep["dataset"],"source_path":str(source),"sha256":ep["sha256"],"validation":"ok"}


def build_datasets():
    parents={name:json.loads((dataset_path(name,MODES[0])/"meta/arx.json").read_text()) for name in TASKS}
    episodes=[ep for name in TASKS for ep in parents[name]["episodes"]]
    assert [len(parents[name]["episodes"]) for name in TASKS]==[52,53]
    with ThreadPoolExecutor(max_workers=8) as pool:
        proofs=list(pool.map(verify_source,episodes))
    save(EVIDENCE/"source_integrity.json",{"validation":"ok","episodes":proofs,
        "video_validation":"Hashes match the previously exhaustive source/video RGB decoding proofs; output videos are identical hardlinks."})
    counts={name:parents[name]["anchors"] for name in TASKS}
    total=sum(ep["output_frames"] for ep in episodes);anchors=sum(counts.values())
    for mode in MODES:
        root=dataset_path(NAME,mode);meta=root/"meta";evidence=EVIDENCE/profile(NAME,mode)
        if (evidence/"ready.json").exists():continue
        (meta/"episodes/chunk-000").mkdir(parents=True,exist_ok=True)
        mapped=[];ep_rows=[];offset=0;vectors={"observation.state":[],"action":[]}
        parent_provenance=[]
        for name in TASKS:
            source_root=dataset_path(name,mode)
            validate_calibrated_contract(source_root,mode)
            parent_provenance.append({"dataset":name,"path":str(source_root),"manifest_sha256":sha256(source_root/"meta/arx.json")})
        for number,old in enumerate(episodes):
            name=old["dataset"];local=old["episode_index"];size=old["output_frames"]
            source_root=dataset_path(name,mode);task_index=list(TASKS).index(name)
            table=pq.read_table(source_root/f"data/chunk-000/file-{local:03d}.parquet")
            with h5py.File(old["source_path"],"r") as source:
                _,state,action=labels(*(source[k][()] for k in ("observations/qpos","observations/eef","action_poscmd")),
                    [old["sides"][s]["baseline"] for s in ("left","right")],mode)
            for key,expected in (("observation.state",state),("action",action)):
                np.testing.assert_array_equal(np.asarray(table[key].to_pylist(),dtype=np.float32),expected)
                vectors[key].append(expected)
            assert len(table)==size
            table=reindex_table(table,number,offset,task_index)
            target=root/f"data/chunk-000/file-{number:03d}.parquet";target.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(table,target)
            ep=copy.deepcopy(old);ep.update(episode_index=number,source_derived_episode_index=local,
                task_index=task_index,task_instruction=TASKS[name],artifacts={})
            ep["artifacts"][str(target.relative_to(DATA))]=sha256(target)
            item={"episode_index":number,"tasks":[TASKS[name]],"length":size,"data/chunk_index":0,
                "data/file_index":number,"dataset_from_index":offset,"dataset_to_index":offset+size,
                "meta/episodes/chunk_index":0,"meta/episodes/file_index":0}
            for camera in CAMERAS:
                prefix=f"videos/observation.images.{camera}"
                video=source_root/f"{prefix}/chunk-000/file-{local:03d}.mp4"
                first=dataset_path(name,MODES[0])/f"{prefix}/chunk-000/file-{local:03d}.mp4"
                assert os.path.samefile(video,first), "source modes must share validated video bytes"
                dest=root/f"{prefix}/chunk-000/file-{number:03d}.mp4";dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert os.path.samefile(dest,video)
                else:os.link(video,dest)
                ep["artifacts"][str(dest.relative_to(DATA))]=old["artifacts"][str(first.relative_to(DATA))]
                item.update({prefix+"/chunk_index":0,prefix+"/file_index":number,prefix+"/from_timestamp":0.,prefix+"/to_timestamp":size/30})
            mapped.append(ep);ep_rows.append(item);offset+=size
        pq.write_table(pa.Table.from_pylist(ep_rows),meta/"episodes/chunk-000/file-000.parquet")
        pd.DataFrame({"task_index":[0,1]},index=list(TASKS.values())).to_parquet(meta/"tasks.parquet")
        native_stats={}
        for key,blocks in vectors.items():
            arr=np.concatenate(blocks).astype(np.float64)
            native_stats[key]={"min":arr.min(0).tolist(),"max":arr.max(0).tolist(),"mean":arr.mean(0).tolist(),"std":arr.std(0).tolist(),"count":[total]}
        save(meta/"stats.json",native_stats)
        info=json.loads((dataset_path("Blue",mode)/"meta/info.json").read_text())
        info.update(total_episodes=105,total_frames=total,total_tasks=2,splits={"train":"0:105"})
        save(meta/"info.json",info)
        manifest={**contract(mode),"dataset":NAME,"tasks":TASKS,"total_episodes":105,"total_frames":total,
            "anchors":anchors,"episodes":mapped,"parent_datasets":parent_provenance,"validation_passed":True,
            "sampling":"uniform over all valid windows; no task oversampling","task_anchors":counts,
            "task_sampling_fractions":{k:v/anchors for k,v in counts.items()},
            "all_component_labels_checked":True,"source_hashes_rechecked":True,
            "calibration_parameters":parents["Blue"]["calibration_parameters"],
            "calibration_window_definition":parents["Blue"]["calibration_window_definition"],
            "video_encoding":parents["Blue"]["video_encoding"],
            "video_validation":"Identical hardlinks to hash-rechecked exhaustive v1 source/video pixel proofs"}
        save(meta/"arx.json",manifest);save(evidence/"source_manifest.json",manifest)
        save(evidence/"conversion_validation.json",{"validation":"ok","episodes":105,"frames":total,"all_labels":True,
            "all_source_sha256":True,"pixel_exact":True,"inherited_complete_image_decode_proofs_verified":True,
            "task_index_mapping":{"0":"Blue","1":"T"},"parent_data_modified":False})
        save(evidence/"converted.json",{"dataset":str(root),"profile":profile(NAME,mode),"episodes":105,"frames":total,
            "anchors":anchors,"task_anchors":counts,"expected_batches_per_rank":((anchors+15)//16)//8,
            "expected_vla_epoch":10000/(((anchors+15)//16)//8)})
        print(mode,"105 episodes merged and labels verified",flush=True)
    save(EVIDENCE/"calibration_audit.json",{"version":"arx-open-baseline-v1","episodes":episodes})


def prepare_mode(mode):
    evidence=EVIDENCE/profile(NAME,mode)
    if (evidence/"ready.json").exists():return
    env=dict(os.environ,PYTHONPATH="src:.",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    python=str(REPO/".venv/bin/python")
    def run(stage,args):
        with (evidence/(stage+".log")).open("w") as stream:
            subprocess.run([python,*map(str,args)],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    body="arx_calibrated_eef_v1" if mode=="eef-vr" else "arx_calibrated_joint_v1"
    run("statistics",["scripts/norm_stats/compute_unified_ft_stats.py","--body",body,"--repos",dataset_path(NAME,mode),
        "--action-horizon",30,"--positive-labels","--negative-labels","--partials-dir",evidence/"stats"])
    stats=REPO/"configs"/config_name(NAME,mode)/"norm_stats.json"
    run("statistics_merge",["scripts/norm_stats/merge_stats.py","--partials",evidence/"stats","--out",stats])
    result=json.loads(stats.read_text());result["calibrated_provenance"]={"dataset":NAME,"contract":contract(mode),
        "source_manifest_sha256":sha256(evidence/"source_manifest.json")};save(stats,result)
    run("dataloader_verification",["scripts/verify_blue_t_dataloader.py",NAME,mode,"--evidence-root",EVIDENCE])
    print(mode,"mixed statistics and DataLoader verification ready",flush=True)


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    with (EVIDENCE/"preparation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build_datasets()
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(prepare_mode,MODES))


if __name__=="__main__":main()
