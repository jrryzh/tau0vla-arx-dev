#!/usr/bin/env python3
"""Ensure GOP2 random access while preserving every source-validated RGB pixel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import av
from prepare_blue_t_training import EVIDENCE,DATA,MODES,TASKS,CAMERAS,dataset_path,profile,save,sha256


def optimize_episode(row):
    name,n=row["dataset"],row["episode_index"]
    for camera in CAMERAS:
        relative=f"videos/observation.images.{camera}/chunk-000/file-{n:03d}.mp4"
        target=dataset_path(name,MODES[0]) / relative
        if row["videos"][camera].get("max_gop")==2:
            assert sha256(target)==row["artifacts"][str(target.relative_to(DATA))]
            continue
        temporary=target.with_suffix('.gop2.mp4')
        with av.open(str(target)) as source, av.open(str(temporary),'w') as dest:
            stream=dest.add_stream('libx264rgb',30,options={'crf':'0','preset':'fast','threads':'1','g':'2'})
            stream.width,stream.height,stream.pix_fmt=640,480,'rgb24'
            source.streams.video[0].codec_context.thread_count=1
            for frame in source.decode(video=0):
                rgb=frame.to_ndarray(format='rgb24')
                dest.mux(stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')))
            dest.mux(stream.encode())
        digest=hashlib.sha256();count=0;last_keyframe=0
        with av.open(str(temporary)) as container:
            assert container.streams.video[0].average_rate==30
            for i,frame in enumerate(container.decode(video=0)):
                if frame.key_frame: last_keyframe=i
                assert i-last_keyframe<2
                assert frame.width==640 and frame.height==480
                assert abs(float(frame.pts*frame.time_base)-i/30)<1e-5
                digest.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
        assert count==row['output_frames'] and digest.hexdigest()==row['videos'][camera]['source_selected_rgb_sha256']
        temporary.replace(target)
        checksum=sha256(target)
        for mode in MODES[1:]:
            linked=dataset_path(name,mode) / relative
            linked.unlink();os.link(target,linked)
        for mode in MODES:
            row['artifacts'][str((dataset_path(name,mode) / relative).relative_to(DATA))]=checksum
        row['videos'][camera]['max_gop']=2
        row['videos'][camera]['gop2_rgb_sha256']=digest.hexdigest()
    save(EVIDENCE / name / f'episode_{n:03d}.json',row)
    print(name,n,'lossless GOP2 validated',flush=True)
    return row


def optimize(name,workers=32):
    marker=EVIDENCE / name / 'gop2_complete.json'
    if marker.exists(): return
    rows=json.loads((dataset_path(name,MODES[0]) / 'meta/arx.json').read_text())['episodes']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(optimize_episode,rows))
    for mode in MODES:
        path=dataset_path(name,mode) / 'meta/arx.json'
        manifest=json.loads(path.read_text())
        manifest['episodes']=rows
        manifest['video_encoding']='lossless libx264rgb crf=0 GOP2; all decoded pixels equal selected source pixels'
        save(path,manifest)
        save(EVIDENCE / profile(name,mode) / 'source_manifest.json',manifest)
    save(marker,{'episodes':len(rows),'max_gop':2,'all_pixels_revalidated':True})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('name',choices=TASKS)
    p.add_argument('--workers',type=int,default=32)
    args=p.parse_args()
    optimize(args.name,args.workers)
