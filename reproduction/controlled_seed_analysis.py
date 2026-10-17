"""Seed-block analysis of corrected controlled decisions; no controller execution."""
from pathlib import Path
import hashlib
import json
import os
from collections import Counter
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=Path(os.environ.get('ICTC_RESEARCH_ROOT', str(ROOT))).resolve()
INPUT=Path(os.environ.get('ICTC_CONTROLLED_INPUT', str(ROOT/'experiments/controlled/controller-heldout-repair-001'))).resolve()
OUT=Path(os.environ.get('ICTC_CONTROLLED_OUTPUT', str(ROOT/'analysis/controlled-seed-correction'))).resolve()
SEEDS=list(range(70000,70020))
BOOT_SEED=100901
REPS=10000

def sha(p):
    return hashlib.file_digest(p.open('rb'),'sha256').hexdigest().upper()

def read(name):
    return [json.loads(s) for s in (INPUT/name).read_text().splitlines()]

def holm(values):
    out={}; previous=0
    for i,(k,p) in enumerate(sorted(values.items(),key=lambda x:x[1])):
        previous=max(previous,min(1.,(len(values)-i)*p));out[k]=previous
    return out

def main():
    OUT.mkdir(exist_ok=False)
    input_names=('ablation_decisions.jsonl','ablation_scores.jsonl','baseline_b0_oracle_decisions.jsonl',
                 'baseline_b1_b8_decisions.jsonl','baseline_scores.jsonl','scoring_audit.json')
    input_hashes={name:sha(INPUT/name) for name in input_names}
    groups={}
    for name,idkey in [('ablation_decisions.jsonl','ablation_id'),('ablation_scores.jsonl','ablation_id'),('baseline_b0_oracle_decisions.jsonl','baseline_id'),('baseline_b1_b8_decisions.jsonl','baseline_id'),('baseline_scores.jsonl','baseline_id')]:
        for r in read(name):
            method=r[idkey]+(':'+r['configuration_id'] if idkey=='ablation_id' else '')
            row=groups.setdefault(method,{}).setdefault(r['pair_id'],{})
            row.update(r)
    ref=groups['A0:default']
    assert len(ref)==1320
    assert Counter(int(k.rsplit('-',1)[1]) for k in ref)==Counter({s:66 for s in SEEDS})
    binary={'false_admissions':'false_admission','latency_nonimprovement':'harmful_action','unnecessary_deferrals':'unnecessary_deferral'}
    metrics=[*binary,'admission_fraction','selected_latency_ms','selected_packet_loss_ratio','selected_jitter_ms','selected_deadline_violation_rate','telemetry_bytes']
    draws=np.random.default_rng(BOOT_SEED).integers(0,20,(REPS,20))
    def value(row,metric):
        if metric=='admission_fraction': return float(row.get('executed_reroute',row.get('selected_route_changes')==1))
        x=row.get(binary.get(metric,metric))
        return None if x is None else float(x)
    comparisons={}; block_records=[]
    for method,rows in sorted(groups.items()):
        assert set(rows)==set(ref),method
        if method=='A0:default':continue
        result={}
        for metric in metrics:
            sums=[];counts=[];ls=[];rs=[]
            for seed in SEEDS:
                pairs=[(value(ref[k],metric),value(rows[k],metric)) for k in sorted(ref) if int(k.rsplit('-',1)[1])==seed]
                pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
                differences=np.array([a-b for a,b in pairs],dtype=float)
                sums.append(float(differences.sum()));counts.append(len(pairs))
                ls.extend(a for a,b in pairs);rs.extend(b for a,b in pairs)
                block_records.append({'method':method,'metric':metric,'seed':seed,'paired_count':len(pairs),'difference_sum':float(differences.sum()),'mean_difference':float(differences.mean()) if len(pairs) else None})
            sums=np.array(sums);counts=np.array(counts)
            if counts.sum()==0:continue
            denominators=counts[draws].sum(axis=1)
            assert (denominators>0).all()
            samples=sums[draws].sum(axis=1)/denominators
            means=np.divide(sums,counts,out=np.zeros(20),where=counts>0)
            pos=int((means>1e-12).sum());neg=int((means< -1e-12).sum())
            p=float(binomtest(pos,pos+neg,.5).pvalue) if pos+neg else 1.
            result[metric]={'paired_count':int(counts.sum()),'independent_blocks':int((counts>0).sum()),'reference_mean':float(np.mean(ls)),'comparator_mean':float(np.mean(rs)),'difference':float(sums.sum()/counts.sum()),'seed_bootstrap_95_ci':np.quantile(samples,[.025,.975]).tolist(),'exact_seed_sign_p':p,'positive_blocks':pos,'negative_blocks':neg,'zero_blocks':int((counts>0).sum())-pos-neg,'paired_count_per_block':counts.tolist()}
        comparisons[method]=result
    baselines=[f'B{i}' for i in range(9)]
    for metric in metrics:
        adjusted=holm({m:comparisons[m][metric]['exact_seed_sign_p'] for m in baselines if metric in comparisons[m]})
        for m,p in adjusted.items():comparisons[m][metric]['holm_p']=p
    result={'amendment':'Post-review temporal and uncertainty correction. Foreground begins at tick 6 after the background-only prelude and tick-5 acquisition cutoff. Training uncertainty uses seed-cluster CR1 influence scores. Not an untouched confirmation; original records remain preserved.',
      'independent_unit':'20 seeds, each containing 66 regime/load/condition cells. Paired methods and all cells remain together.',
      'dependence_evidence':'schedule.py derives foreground, cross:path and telemetry:path RNGs only from seed and stream name, not condition, regime or load. Same-seed scenarios reuse streams.',
      'bootstrap':'10000 percentile complete-seed resamples; pooled paired sum/count within each resample; zero-denominator replicates forbidden',
      'bootstrap_seed':BOOT_SEED,'test':'Two-sided exact sign test on nonzero seed-mean paired differences; tolerance 1e-12; does not assume symmetric magnitudes. Tests directional seed effects, not a null of equal pooled means.',
      'multiplicity':'Holm separately for each endpoint over nine reference-versus-B0--B8 contrasts. Ablation p-values descriptive and unadjusted. Confidence intervals nominal.',
      'source_hashes':input_hashes,
      'pair_count':len(ref),'seeds':SEEDS,'comparisons':comparisons}
    (OUT/'corrected_inference.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'seed_contrasts.json').write_text(json.dumps(block_records,indent=2)+'\n')
    # Compact portable record set permits an independent recomputation without the native archive.
    packed={m:{k:{'seed':int(k.rsplit('-',1)[1]),**{v:value(r,v) for v in metrics}} for k,r in rows.items()} for m,rows in groups.items()}
    (OUT/'retained_endpoint_records.json').write_text(json.dumps(packed,separators=(',',':'))+'\n')
    for metric in ('false_admissions','selected_latency_ms','telemetry_bytes'):
        vals=[comparisons[m][metric] for m in baselines]
        print(metric,json.dumps({'difference_range':[min(r['difference'] for r in vals),max(r['difference'] for r in vals)],'min_ci_lower':min(r['seed_bootstrap_95_ci'][0] for r in vals),'max_ci_upper':max(r['seed_bootstrap_95_ci'][1] for r in vals),'max_p':max(r['exact_seed_sign_p'] for r in vals),'max_holm':max(r['holm_p'] for r in vals)}))
    for method in ('A1:default','A2:default','A5:default','A6:default','A7:default'):
        if method in comparisons:print(method,json.dumps(comparisons[method]['false_admissions']))

if __name__=='__main__':main()
