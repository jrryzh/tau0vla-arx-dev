import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from manage_blue_t_resources import choose, reassessed_formal_deficit, PROFILES, MIXED_PROFILES
from manage_0905_resources import GROUP, OWNER
from report_blue_t_campaign import advance_acceptance


def job(name,status,created=1,owner=OWNER,priority=10):
    return {"name":name,"job_id":name,"status":status,"created_at":created,"created_by":{"id":owner},
        "logic_compute_group_id":GROUP,"node_count":2 if status=="job_running" else 0,"priority":priority}


def test_release_only_authorized_filler_for_actual_deficit():
    ours=job('arx-0907-blue-joint-vr-h200-smoke','job_queuing',created=10)
    fill=job('qwen35_vla_fill_16g_old','job_running')
    assert choose([ours,fill],0)==fill
    assert choose([ours,fill],2) is None
    assert choose([fill],0) is None
    unrelated=job('real-training','job_running')
    assert choose([ours,unrelated],0) is None
    wrong_owner=job('qwen35_vla_fill_16g_old','job_running',owner='someone-else')
    assert choose([ours,wrong_owner],0) is None
    competitor=job('real-training','job_queuing',created=2)
    assert choose([ours,fill,competitor],0) is None


def test_checkpoint_acceptance_needs_two_later_checks_and_rechecks_after_recovery():
    a={"checkpoint_validated":True,"growth_checks":[],"baseline_step":500,"validated_at":100}
    advance_acceptance(a,step=510,now=110,job_id='a',healthy=True)
    assert not a['accepted'] and len(a['growth_checks'])==0
    advance_acceptance(a,step=520,now=140,job_id='a',healthy=True)
    assert not a['accepted'] and len(a['growth_checks'])==1
    advance_acceptance(a,step=520,now=180,job_id='a',healthy=True)
    assert not a['accepted'] and len(a['growth_checks'])==1
    advance_acceptance(a,step=540,now=200,job_id='a',healthy=True)
    assert a['accepted']
    advance_acceptance(a,step=540,now=240,job_id='a',healthy=False,reset=True)
    assert not a['accepted'] and not a['growth_checks']
    advance_acceptance(a,step=510,now=300,job_id='b',healthy=True)
    assert not a['accepted'] and not a['growth_checks']
    advance_acceptance(a,step=520,now=340,job_id='b',healthy=True)
    advance_acceptance(a,step=530,now=380,job_id='b',healthy=True)
    assert a['accepted']


def test_persistent_formal_deficit_uses_actual_allocation_after_releases_settle():
    jobs=[job('arx-'+p+'-formal','job_running',created=1000) for p in PROFILES[:5]]
    queued=job('arx-'+PROFILES[5]+'-formal','job_queuing',created=1000)
    jobs.append(queued)
    ids=[j['job_id'] for j in jobs]
    assert reassessed_formal_deficit(jobs,ids,0,[],1000)['missing_nodes']==2
    assert reassessed_formal_deficit(jobs,ids,2,[],1000) is None
    assert reassessed_formal_deficit(jobs,ids,0,[],60) is None
    smoke=job('arx-'+PROFILES[0]+'-smoke','job_running')
    assert reassessed_formal_deficit(jobs+[smoke],ids,0,[],1000) is None
    recent=[{'time':'1970-01-01T00:16:00+00:00'}]
    assert reassessed_formal_deficit(jobs,ids,0,recent,1000) is None


def test_mixed_and_separate_campaigns_only_release_for_their_own_profiles():
    fill=job('qwen35_vla_fill_16g_old','job_running')
    mixed=job('arx-'+MIXED_PROFILES[0]+'-smoke','job_queuing',created=1000)
    separate=job('arx-'+PROFILES[0]+'-formal','job_running')
    assert choose([mixed,fill,separate],0) is None
    assert choose([mixed,fill,separate],0,MIXED_PROFILES)==fill
    assert choose([job('arx-'+PROFILES[0]+'-formal','job_queuing'),fill],0,MIXED_PROFILES) is None
    formal=[job('arx-'+p+'-formal','job_running',created=1000) for p in MIXED_PROFILES[:2]]
    formal.append(job('arx-'+MIXED_PROFILES[2]+'-formal','job_queuing',created=1000))
    result=reassessed_formal_deficit(formal+[separate],[j['job_id'] for j in formal],0,[],1000,MIXED_PROFILES)
    assert result['missing_nodes']==2 and result['allocated_formal_nodes']==4


def test_mixed_reindex_preserves_vectors_and_local_time():
    import pyarrow as pa
    from prepare_bluet_mixed_training import reindex_table
    table=pa.table({'observation.state':[[1.,2.],[3.,4.]],'action':[[5.,6.],[7.,8.]],
        'episode_index':[0,0],'index':[0,1],'task_index':[0,0],'frame_index':[0,1],'timestamp':[0.,1/30]})
    mixed=reindex_table(table,52,36494,1)
    for key in ('observation.state','action','frame_index','timestamp'):assert mixed[key].equals(table[key])
    assert mixed['episode_index'].to_pylist()==[52,52]
    assert mixed['task_index'].to_pylist()==[1,1]
    assert mixed['index'].to_pylist()==[36494,36495]
    assert table['task_index'].to_pylist()==[0,0]
