"""Isolated fixed-input memory diagnostic; no fitting, outcomes, or native runs."""
from pathlib import Path
import argparse
import ctypes
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('ICTC_RESEARCH_ROOT', str(ROOT))).resolve()
OUTPUT = ROOT / 'analysis/process-memory'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def counters():
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),
                    ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
                    ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    value = Counters(); value.cb = ctypes.sizeof(value)
    handle = ctypes.windll.kernel32.GetCurrentProcess
    handle.restype = ctypes.c_void_p
    get = ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get.restype = ctypes.c_int
    if not get(handle(),ctypes.byref(value),value.cb):
        raise ctypes.WinError()
    return {'working_set_bytes':int(value.WorkingSetSize),'private_commit_bytes':int(value.PagefileUsage),
            'peak_working_set_bytes':int(value.PeakWorkingSetSize),'peak_private_commit_bytes':int(value.PeakPagefileUsage)}

def worker(kind, admission):
    sys.path[:0] = [str(EVIDENCE/'code'),str(EVIDENCE/'code/src')]
    import joblib
    # Import dependencies identically in every process before measuring the baseline.
    import sklearn.ensemble
    from ictc.ns3.n12 import N12Scenario
    from revision_controller import context, admit, learned_proposal, fluid_proposal
    from benchmark_revision import baseline
    inputs = read(OUTPUT/'input.json')
    scenario = N12Scenario(**inputs['scenario'])
    raw = EVIDENCE/inputs['raw_relative']
    gc.collect()
    before_load = counters()
    model = joblib.load(OUTPUT/'selected-tree.joblib') if kind=='Learned' else None
    risk = read(OUTPUT/f'{kind.lower()}-risk.json') if admission else None
    threshold = inputs['thresholds'][kind]
    def operation():
        if kind=='Greedy' and not admission:
            return baseline(raw,scenario)
        ctx = context(raw,scenario)
        proposed = ctx['greedy_action'] if kind=='Greedy' else learned_proposal(scenario,ctx,model) if kind=='Learned' else fluid_proposal(scenario,ctx)
        return admit(scenario,ctx,proposed,risk,threshold) if admission else proposed
    after_load = counters()
    for _ in range(50):
        assert operation()=='route2'
    gc.collect()
    after_warmup = counters()
    for _ in range(1000):
        assert operation()=='route2'
    gc.collect()
    after_calls = counters()
    return {'method':kind,'admission':admission,'pid':os.getpid(),'before_model_load':before_load,
            'after_model_load':after_load,'after_50_warmup':after_warmup,'after_1000_calls':after_calls,
            'selected_action':'route2','native_runs':0,'model_fits':0,'outcome_files_read':0}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--admission',action='store_true')
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker,args.admission)));return
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    import joblib
    matrix=read(EVIDENCE/'configs/abilene-test-matrix.json')
    scope=read(EVIDENCE/'analysis/controller_overhead_active/benchmark_scope.json')
    scenario=next(s for s in matrix['scenarios'] if s['scenario_id']==scope['scenario_id'])
    relative=f"experiments/abilene/test/groups/{scope['scenario_id']}/probe-route0/raw"
    assert sha(EVIDENCE/relative/'telemetry.csv').upper()==scope['telemetry_sha256'].upper()
    selected=read(EVIDENCE/'analysis/models/abilene/selection.json')
    bundle=joblib.load(EVIDENCE/'analysis/models/abilene/bundle.joblib')
    depth=selected['depth']
    joblib.dump(bundle['proposers'][depth],OUTPUT/'selected-tree.joblib')
    risk={'Greedy':read(EVIDENCE/'code/configs/experiments/n13_selected_policy.json')['controller_thresholds']['risk_model'],
          'Learned':bundle['risk'][f'learned{depth}'],'Fluid':bundle['risk']['fluid']}
    for name,model in risk.items():
        (OUTPUT/f'{name.lower()}-risk.json').write_text(json.dumps(model,indent=2)+'\n',encoding='utf-8')
    inputs={'scenario':scenario,'raw_relative':relative,'telemetry_sha256':scope['telemetry_sha256'],
            'bundle_sha256':sha(EVIDENCE/'analysis/models/abilene/bundle.joblib'),
            'thresholds':{'Greedy':0.30,'Learned':selected['thresholds'][f'learned{depth}'],'Fluid':selected['thresholds']['fluid']},
            'selected_tree_sha256':sha(OUTPUT/'selected-tree.joblib')}
    (OUTPUT/'input.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
    records=[]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    for repetition in range(3):
        for kind in ('Greedy','Learned','Fluid'):
            # Counterbalance process order; no paired-memory inference is claimed.
            for admission in ((False,True) if repetition%2==0 else (True,False)):
                command=[sys.executable,str(Path(__file__)), '--worker',kind]+(['--admission'] if admission else [])
                completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,check=True)
                row=json.loads(completed.stdout);row['repetition']=repetition+1
                records.append(row)
                (OUTPUT/f"{kind.lower()}-{'ictc' if admission else 'alone'}-{repetition+1}.json").write_text(json.dumps(row,indent=2)+'\n')
                print(f"Memory diagnostic {len(records)}/18 complete",flush=True)
    groups=[]
    for kind in ('Greedy','Learned','Fluid'):
        for admission in (False,True):
            rows=[r for r in records if r['method']==kind and r['admission']==admission]
            summaries={}
            for key in ('working_set_bytes','private_commit_bytes','peak_working_set_bytes','peak_private_commit_bytes'):
                values=[r['after_1000_calls'][key] for r in rows]
                summaries[key]={'median':statistics.median(values),'minimum':min(values),'maximum':max(values)}
            for key in ('working_set_bytes','private_commit_bytes'):
                values=[r['after_1000_calls'][key]-r['before_model_load'][key] for r in rows]
                summaries['load_and_calls_increment_'+key]={'median':statistics.median(values),'minimum':min(values),'maximum':max(values)}
            groups.append({'method':kind,'admission':admission,'independent_processes':3,'counters':summaries})
    report={'protocol':'Three fresh processes per variant; identical imports; baseline before loading only the required selected model; 50 warm-up calls; 1000 fixed-input calls; GC before snapshots. Counters are whole-process snapshots, not peaks attributable solely to a controller. Cross-variant contrasts are differences between independent medians, not paired increments.',
            'units':'bytes','native_runs':0,'model_fits':0,'outcome_files_read':0,'groups':groups,'records':records}
    (OUTPUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PROCESS_MEMORY_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
