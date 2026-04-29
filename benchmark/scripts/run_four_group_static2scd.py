"""
四组对准精度递进实验:
G1: 普通KF(36-state) @ 单位置静止
G2: Markov(42-state) @ 单位置静止
G3: Markov(42-state) @ 20-position旋转策略
G4: Markov+SCD @ 20-position旋转策略
"""
from __future__ import annotations
import sys, types, time, json, copy, math
from pathlib import Path

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TMP_PSINS_DIR = ROOT / 'tmp_psins_py'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'

for p in [str(ROOT), str(TMP_PSINS_DIR), str(METHOD_DIR), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Load source module
import importlib.util
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

mod = load_mod('fgr_source', str(TMP_PSINS_DIR / 'psins_py' / 'test_calibration_markov_pruned.py'))
np = mod.np

# Load SCD infrastructure from compare_four_methods_shared_noise
_compare_shared = load_mod('_fgr_shared', str(SCRIPTS_DIR / 'compare_four_methods_shared_noise.py'))
_probe_r55 = load_mod('_fgr_r55', str(SCRIPTS_DIR / 'probe_round55_newline.py'))
_compute_r61 = load_mod('_fgr_r61', str(SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'))

R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
SYMMETRIC20_PROBE_FILE = SCRIPTS_DIR / 'probe_ch3_corrected_symmetric20_front2_back11.py'
R61_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round61_h_scd_state20_microtight_commit.py'

BASE_ARW, BASE_VRW, BASE_BI_G, BASE_BI_A = 0.005, 5.0, 0.002, 5.0
NOISE_SCALE = 0.12
SEED = 42
COMPARISON_MODE = 'four_group_static_to_scd_alignment'

GROUP_DISPLAY = {
    'g1': 'G1 普通模型 @ 单位置对准',
    'g2': 'G2 Markov 模型 @ 单位置对准',
    'g3': 'G3 Markov @ 旋转对准策略 (20-position)',
    'g4': 'G4 Markov + SCD @ 旋转对准策略 (20-position)',
}
FOCUS_PARAMS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z']

def make_dataset(att0_deg, paras_raw, noise_scale):
    ts = 0.01
    att0 = np.array([float(x) for x in att0_deg]) * mod.glv.deg
    pos0 = mod.posset(34.0, 0.0, 0.0)
    # Build paras array
    if isinstance(paras_raw, list):
        paras = np.array(paras_raw, dtype=float)
        if paras.shape[1] >= 5:
            paras[:, 4] = paras[:, 4] * mod.glv.deg
        att = mod.attrottt(att0, paras, ts)
    else:
        # Static: just a single attitude for duration_s
        duration_s, att_static = paras_raw
        n = int(duration_s / ts) + 1
        att = np.zeros((n, 4))
        att[:, 0:3] = att_static
        att[:, 3] = np.arange(1, n + 1) * ts

    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)

    arw = BASE_ARW * noise_scale * mod.glv.dpsh
    vrw = BASE_VRW * noise_scale * mod.glv.ugpsHz
    bi_g = BASE_BI_G * noise_scale * mod.glv.dph
    bi_a = BASE_BI_A * noise_scale * mod.glv.ug

    imu_noisy = mod.imuadderr_full(imu_clean, ts, arw=arw, vrw=vrw, bi_g=bi_g, tau_g=300.0, bi_a=bi_a, tau_a=300.0, seed=SEED)

    cfg = {
        'arw_dpsh': BASE_ARW * noise_scale,
        'vrw_ugpsHz': BASE_VRW * noise_scale,
        'bi_g_dph': BASE_BI_G * noise_scale,
        'bi_a_ug': BASE_BI_A * noise_scale,
        'tau_g': 300.0, 'tau_a': 300.0,
        'seed': SEED,
        'base_family': 'round53_round61_shared',
    }
    return {
        'ts': ts, 'pos0': pos0, 'imu_noisy': imu_noisy,
        'bi_g': bi_g, 'bi_a': bi_a,
        'noise_scale': noise_scale,
        'noise_config': cfg,
    }

def compute_payload(clbt, params, variant, method_file, extra):
    param_errors = {}
    pcts = []
    for name, true_v, get_est in params:
        true_f, est_f = float(true_v), float(get_est(clbt))
        ae = abs(true_f - est_f)
        pe = ae / abs(true_f) * 100.0 if abs(true_f) > 1e-15 else 0.0
        param_errors[name] = {'true': true_f, 'est': est_f, 'abs_error': ae, 'pct_error': pe}
        pcts.append(pe)
    arr = np.asarray(pcts, dtype=float)
    overall = {'mean_pct_error': float(np.mean(arr)), 'median_pct_error': float(np.median(arr)), 'max_pct_error': float(np.max(arr))}
    return {
        'variant': variant, 'method_file': method_file,
        'source_file': str(TMP_PSINS_DIR / 'psins_py' / 'test_calibration_markov_pruned.py'),
        'param_order': [n for n, _, _ in params], 'param_errors': param_errors,
        'focus_scale_pct': {p: param_errors[p]['pct_error'] for p in ['dKg_xx','dKg_xy','dKg_yy','dKg_zz','dKa_xx']},
        'lever_guard_pct': {p: param_errors[p]['pct_error'] for p in ['rx_y','ry_z']},
        'overall': overall, 'extra': extra or {},
    }

def run_kf36_static(ds, params):
    print("  G1: KF36 static...", end=" ", flush=True)
    t0 = time.time()
    clbt, *_ = mod.run_calibration(ds['imu_noisy'], ds['pos0'], ds['ts'], n_states=36, label='G1-KF36-STATIC')
    print(f"done {time.time()-t0:.1f}s")
    return compute_payload(clbt, params, 'g1_kf_static', 'source_mod.run_calibration(n_states=36)',
        extra={'comparison_mode':COMPARISON_MODE,'group_key':'g1','noise_scale':ds['noise_scale'],'noise_config':ds['noise_config']})

def run_markov42_static(ds, params):
    print("  G2: Markov42 static...", end=" ", flush=True)
    t0 = time.time()
    clbt, *_ = mod.run_calibration(ds['imu_noisy'], ds['pos0'], ds['ts'], n_states=42,
        bi_g=ds['bi_g'], tau_g=300.0, bi_a=ds['bi_a'], tau_a=300.0, label='G2-MARKOV42-STATIC')
    print(f"done {time.time()-t0:.1f}s")
    return compute_payload(clbt, params, 'g2_markov_static', 'source_mod.run_calibration(n_states=42, markov)',
        extra={'comparison_mode':COMPARISON_MODE,'group_key':'g2','noise_scale':ds['noise_scale'],'noise_config':ds['noise_config']})

def run_markov42_sym20(ds, params):
    print("  G3: Markov42 sym20...", end=" ", flush=True)
    t0 = time.time()
    clbt, *_ = mod.run_calibration(ds['imu_noisy'], ds['pos0'], ds['ts'], n_states=42,
        bi_g=ds['bi_g'], tau_g=300.0, bi_a=ds['bi_a'], tau_a=300.0, label='G3-MARKOV42-SYM20')
    print(f"done {time.time()-t0:.1f}s")
    return compute_payload(clbt, params, 'g3_markov_sym20', 'source_mod.run_calibration(n_states=42, markov) on sym20',
        extra={'comparison_mode':COMPARISON_MODE,'group_key':'g3','noise_scale':ds['noise_scale'],'noise_config':ds['noise_config']})

def run_scd_sym20(ds, params):
    print("  G4: SCD sym20...", end=" ", flush=True)
    t0 = time.time()
    # Build neutral SCD candidate
    neutral_patch = {
        'selected_prior_scale':1.0,'other_scale_prior_scale':1.0,'ka2_prior_scale':1.0,'lever_prior_scale':1.0,
        'selected_q_static_scale':1.0,'selected_q_dynamic_scale':1.0,'selected_q_late_mult':1.0,
        'other_scale_q_scale':1.0,'other_scale_q_late_mult':1.0,'ka2_q_scale':1.0,'lever_q_scale':1.0,
        'static_r_scale':1.0,'dynamic_r_scale':1.0,'late_r_mult':1.0,'late_release_frac':0.58,
        'selected_alpha_floor':1.0,'selected_alpha_span':0.0,'other_scale_alpha':1.0,
        'ka2_alpha':1.0,'lever_alpha':1.0,'markov_alpha':1.0,
        'trust_score_soft':2.1,'trust_cov_soft':0.44,'trust_mix':0.58,
        'state_alpha_mult':{},'state_alpha_add':{},'state_prior_diag_mult':{},
        'state_q_static_mult':{},'state_q_dynamic_mult':{},'state_q_late_mult':{},
    }
    scd_cand = {
        'name':'neutral_markov42_plus_scd_baseline',
        'description':'Controlled neutral Markov+SCD baseline.',
        'iter_patches':{0:copy.deepcopy(neutral_patch),1:copy.deepcopy(neutral_patch)},
        'post_rx_y_mult':1.0,'post_ry_z_mult':1.0,
        'scd':{'mode':'once_per_phase','alpha':0.999,'transition_duration':2.0,'target':'scale_block','bias_to_target':True,'apply_policy_names':['iter2_commit']},
    }
    r53 = load_mod('g4_r53', str(R53_METHOD_FILE))
    patched = _probe_r55._build_patched_method(r53, scd_cand)
    r61_mod = load_mod('g4_r61_run', str(R61_METHOD_FILE))
    result = list(r61_mod._run_internalized_hybrid_scd(
        patched, mod, ds['imu_noisy'], ds['pos0'], ds['ts'],
        bi_g=ds['bi_g'], bi_a=ds['bi_a'], tau_g=300.0, tau_a=300.0,
        label='G4-SCD-SYM20', scd_cfg=scd_cand['scd'],
    ))
    runtime = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    extra_runtime = {}
    if isinstance(runtime, dict):
        extra_runtime = {'schedule_log': runtime.get('schedule_log'), 'feedback_log': runtime.get('feedback_log'), 'scd_log': runtime.get('scd_log')}
    print(f"done {time.time()-t0:.1f}s")
    return compute_payload(result[0], params, 'g4_scd_sym20', 'neutral_markov42_plus_scd_on_sym20',
        extra={'comparison_mode':COMPARISON_MODE,'group_key':'g4','noise_scale':ds['noise_scale'],'noise_config':ds['noise_config'],
               'scd_cfg':copy.deepcopy(scd_cand['scd']),'iter_patches':copy.deepcopy(scd_cand['iter_patches']),
               'runtime_log':extra_runtime})

def fmt_pct(x): return f'{x:.6f}'

def main():
    params = _compute_r61._param_specs(mod)

    # Sym20 trajectory rows (20 positions, uniform 60s each)
    probe_sym20 = load_mod('fgr_sym20p', str(SYMMETRIC20_PROBE_FILE))
    sym20_candidate = probe_sym20.build_symmetric20_candidate(mod)
    sym20_paras_raw = [[
        idx, int(r['axis'][0]), int(r['axis'][1]), int(r['axis'][2]),
        float(r['angle_deg']), float(r['rotation_time_s']), float(r['pre_static_s']), float(r['post_static_s'])
    ] for idx, r in enumerate(sym20_candidate.all_rows, 1)]

    print("=== 四组实验: 单位置 -> 旋转对准 (noise=0.12x) ===\n")

    # Datasets
    ds_static = make_dataset([0.0, 0.0, 0.0], (1200.0, np.array([0.0,0.0,0.0])*mod.glv.deg), NOISE_SCALE)
    ds_sym20 = make_dataset([0.0, 0.0, 0.0], sym20_paras_raw, NOISE_SCALE)

    print(f"Static dataset: {ds_static['imu_noisy'].shape}")
    print(f"Sym20 dataset:  {ds_sym20['imu_noisy'].shape}")
    print()

    payloads = {}
    execution = {}
    paths = {}

    payloads['g1'], execution['g1'], paths['g1'] = run_kf36_static(ds_static, params), 'rerun', 'g1_kf_static'
    payloads['g2'], execution['g2'], paths['g2'] = run_markov42_static(ds_static, params), 'rerun', 'g2_markov_static'
    payloads['g3'], execution['g3'], paths['g3'] = run_markov42_sym20(ds_sym20, params), 'rerun', 'g3_markov_sym20'
    payloads['g4'], execution['g4'], paths['g4'] = run_scd_sym20(ds_sym20, params), 'rerun', 'g4_scd_sym20'

    # Build summary
    group_rows = []
    order = ['g1','g2','g3','g4']
    for gk in order:
        ov = payloads[gk]['overall']
        group_rows.append({'group_key':gk,'display':GROUP_DISPLAY[gk],
            'mean_pct_error':ov['mean_pct_error'],'median_pct_error':ov['median_pct_error'],'max_pct_error':ov['max_pct_error']})

    # Progression steps
    prog = {}
    for m in ['mean_pct_error','median_pct_error','max_pct_error']:
        vals = [r[m] for r in group_rows]
        steps = []
        for i in range(1, len(vals)):
            d = float(vals[i-1] - vals[i])
            steps.append({'from':GROUP_DISPLAY[order[i-1]],'to':GROUP_DISPLAY[order[i]],'delta':d,'improved':d>0})
        prog[m] = {'strict_progression':all(s['improved'] for s in steps),'steps':steps,
            'best_group':min(group_rows,key=lambda x:x[m])['group_key']}

    # All param table
    ap = {}
    for name in payloads['g1']['param_order']:
        ap[name] = {'true':float(payloads['g1']['param_errors'][name]['true']),'groups':{},'ranking':[]}
        ranking = []
        for gk in order:
            p = float(payloads[gk]['param_errors'][name]['pct_error'])
            e = float(payloads[gk]['param_errors'][name]['est'])
            ap[name]['groups'][gk] = {'est':e,'pct_error':p}
            ranking.append({'group_key':gk,'pct_error':p})
        ranking.sort(key=lambda x:x['pct_error'])
        ap[name]['ranking'] = ranking

    # Headline
    mp = prog['mean_pct_error']['strict_progression']
    dp = prog['median_pct_error']['strict_progression']
    xp = prog['max_pct_error']['strict_progression']
    if mp and dp and xp:
        headline = '四组在 mean/median/max 三个整体指标上均呈现严格递进改善。'
    elif mp:
        headline = '四组在 mean 指标上形成清晰递进；median/max 不一定严格单调。'
    else:
        headline = '这四组不构成完全严格单调递进；需按 mean/median/max 分开看，结合关键参数判断。'
    best_mean = GROUP_DISPLAY[prog['mean_pct_error']['best_group']]
    best_med = GROUP_DISPLAY[prog['median_pct_error']['best_group']]
    best_max = GROUP_DISPLAY[prog['max_pct_error']['best_group']]
    interp = f'当前 best-mean 是 {best_mean}，best-median 是 {best_med}，best-max 是 {best_max}。'

    cfg = ds_static['noise_config']
    # Build report
    lines = []
    lines.append('# 四组对准精度递进对照：单位置 -> 旋转对准')
    lines.append('')
    lines.append(f'noise_scale = **{NOISE_SCALE}**, arw={cfg["arw_dpsh"]:.6f} dpsh, vrw={cfg["vrw_ugpsHz"]:.2f} ugpsHz')
    lines.append(f'bi_g={cfg["bi_g_dph"]:.6f} dph, bi_a={cfg["bi_a_ug"]:.1f} ug, tau=300s, seed={SEED}')
    lines.append('')
    lines.append('')
    lines.append('')
    lines.append('| 组别 | mean% | median% | max% | Δmean vs prev | Δmedian vs prev | Δmax vs prev |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    prev = None
    for r in group_rows:
        if prev is None:
            dm=dmed=dx='—'
        else:
            dm = f'{prev["mean_pct_error"]-r["mean_pct_error"]:+.6f}'
            dmed = f'{prev["median_pct_error"]-r["median_pct_error"]:+.6f}'
            dx = f'{prev["max_pct_error"]-r["max_pct_error"]:+.6f}'
        lines.append(f'| {r["display"]} | {fmt_pct(r["mean_pct_error"])} | {fmt_pct(r["median_pct_error"])} | {fmt_pct(r["max_pct_error"])} | {dm} | {dmed} | {dx} |')
        prev = r
    lines.append('')
    lines.append('### 递进判断')
    for m,label in [('mean_pct_error','mean'),('median_pct_error','median'),('max_pct_error','max')]:
        pm = prog[m]
        v = '严格递进' if pm['strict_progression'] else '非严格单调'
        lines.append(f'- **{label}**：{v}；最优：{GROUP_DISPLAY[pm["best_group"]]}')
        for s in pm['steps']:
            st = '改善' if s['improved'] else '退化'
            lines.append(f'  - {s["from"]} → {s["to"]}：{st} {s["delta"]:+.6f}')
    lines.append('')
    lines.append('### 关键参数（越低越好）')
    lines.append('| 参数 | G1 KF静态 | G2 Markov静态 | G3 Markov旋转 | G4 SCD旋转 | best |')
    lines.append('|---|---:|---:|---:|---:|---|')
    for name in FOCUS_PARAMS:
        best = GROUP_DISPLAY[ap[name]['ranking'][0]['group_key']]
        lines.append(f'| {name} | {fmt_pct(ap[name]["groups"]["g1"]["pct_error"])} | {fmt_pct(ap[name]["groups"]["g2"]["pct_error"])} | {fmt_pct(ap[name]["groups"]["g3"]["pct_error"])} | {fmt_pct(ap[name]["groups"]["g4"]["pct_error"])} | {best} |')
    lines.append('')
    lines.append(f'主线结论：{headline}')
    lines.append(f'{interp}')
    lines.append('')

    # Save files
    import datetime
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    compare_json = RESULTS_DIR / f'compare_four_group_static_to_scd_noise0p12.json'
    report_md = ROOT / 'reports' / f'psins_four_group_static_to_scd_{now}_noise0p12.md'

    compare_json.write_text(json.dumps({
        'experiment':'four_group_static_to_scd_alignment','comparison_mode':COMPARISON_MODE,
        'noise_scale':NOISE_SCALE,'noise_config':cfg,'execution':execution,
        'group_rows':group_rows,'progression_metrics':prog,
        'all_params':{'param_order':payloads['g1']['param_order'],'table':ap},
        'headline':headline,'interpretation':interp,
        'files':{'compare_json':str(compare_json),'report_md':str(report_md),'run_jsons':{g:paths[g] for g in order}},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    report_md.write_text('\n'.join(lines), encoding='utf-8')

    # Save payloads
    for gk in order:
        p = RESULTS_DIR / f'{paths[gk]}_noise0p12_param_errors.json'
        p.write_text(json.dumps(payloads[gk], ensure_ascii=False, indent=2), encoding='utf-8')
        paths[gk] = str(p)

    print('\n=== 四组对照汇总 ===')
    for r in group_rows:
        print(f'{r["display"]:50s}  mean={r["mean_pct_error"]:.6f}  median={r["median_pct_error"]:.6f}  max={r["max_pct_error"]:.6f}')
    print(f'\n{headline}')
    print(f'{interp}')
    print(f'\ncompare_json: {compare_json}')
    print(f'report_md: {report_md}')
    print(f'\n__RESULT_JSON__='+json.dumps({'group_rows':group_rows,'progression_metrics':prog},ensure_ascii=False))

if __name__ == '__main__':
    main()
