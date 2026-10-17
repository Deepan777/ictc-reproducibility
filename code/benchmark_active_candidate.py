"""Additional actual candidate-active microbenchmark; outcomes are not read."""
from __future__ import annotations
import json
import joblib
from ictc.ns3.n12 import N12Scenario, sha256
from benchmark_revision import measurement, baseline, process_memory
from revision_controller import context, admit, learned_proposal, fluid_proposal
from run_revision_native import ROOT, read, new
from analyze_revision import csv_new


def main() -> None:
    output=ROOT/'analysis/controller_overhead_active'
    if output.exists():raise FileExistsError(output)
    matrix=read(ROOT/'configs/abilene-test-matrix.json')
    bundle=joblib.load(ROOT/'analysis/models/abilene/bundle.joblib')
    selection=read(ROOT/'analysis/models/abilene/selection.json')
    original=read(ROOT/'code/configs/experiments/n13_selected_policy.json')['controller_thresholds']['risk_model']
    depth=selection['depth']; selected=None; rejected=None
    for config in matrix['scenarios']:
        scenario=N12Scenario(**config)
        raw=ROOT/f'experiments/abilene/test/groups/{scenario.scenario_id}/probe-route0/raw'
        ctx=context(raw,scenario)
        if rejected is None and not ctx['quality_ready']:
            rejected=(scenario,raw)
        if ctx['quality_ready'] and ctx['greedy_action']!='route0' and learned_proposal(scenario,ctx,bundle['proposers'][depth])!='route0' and fluid_proposal(scenario,ctx)!='route0':
            selected=(scenario,raw,ctx);break
    if selected is None:raise RuntimeError('No candidate-active received-telemetry condition')
    scenario,raw,ctx=selected
    def online(kind:str, assurance:bool) -> str:
        value=context(raw,scenario)
        proposer=value['greedy_action'] if kind=='Greedy' else learned_proposal(scenario,value,bundle['proposers'][depth]) if kind=='Learned' else fluid_proposal(scenario,value)
        model=original if kind=='Greedy' else bundle['risk'][f'learned{depth}' if kind=='Learned' else 'fluid']
        threshold=0.30 if kind=='Greedy' else selection['thresholds'][f'learned{depth}' if kind=='Learned' else 'fluid']
        return admit(scenario,value,proposer,model,threshold) if assurance else proposer
    operations=[('Greedy active candidate including parsing',lambda:baseline(raw,scenario)),
                ('ICTC active candidate including parsing',lambda:online('Greedy',True)),
                ('Learned active candidate including parsing',lambda:online('Learned',False)),
                ('Learned active candidate with ICTC',lambda:online('Learned',True)),
                ('Fluid active candidate including parsing',lambda:online('Fluid',False)),
                ('Fluid active candidate with ICTC',lambda:online('Fluid',True))]
    results,samples=[],[]
    for name,operation in operations:
        result,raw_samples=measurement(name,operation)
        results.append(result);samples.extend(raw_samples)
    output.mkdir(parents=True)
    csv_new(output/'component_latency.csv',results)
    csv_new(output/'raw_timing_samples.csv',samples)
    new(output/'benchmark_scope.json',{'scenario_id':scenario.scenario_id,'telemetry_sha256':sha256(raw/'telemetry.csv'),
        'selection_rule':'first frozen matrix entry with quality-valid nonincumbent Greedy, learned and fluid proposals; selection uses received telemetry and fixed models only',
        'outcome_files_read':0,'warmup':50,'repetitions':1000,'proposals':{kind:online(kind,False) for kind in ('Greedy','Learned','Fluid')},
        'admitted_actions':{kind:online(kind,True) for kind in ('Greedy','Learned','Fluid')},'snapshot':ctx['snapshot'],
        'process_memory':process_memory(),'scope':'host-local full candidate path including CSV data access and preprocessing; no live actuation',
        'initial_measurement_scope':'first frozen matrix entry proposed the incumbent; original timings retained separately',
        'tails':'repeated fixed-input processing variability, not across-scenario latency distributions'})
    print('ACTIVE_CANDIDATE_PATH_BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':main()
