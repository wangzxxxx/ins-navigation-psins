import json
from pathlib import Path
import importlib.util
import sys
import numpy as np

ROOT = Path('/root/.openclaw/workspace')
METHOD = ROOT / 'psins_method_bench' / 'methods' / 'markov' / 'method_42state_gm1_round25_pairgate_rankedtrim.py'
SRC = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
OUT = ROOT / 'psins_method_bench' / 'results' / 'R25_markov_42state_gm1_round25_pairgate_rankedtrim_param_errors.json'


def load(path, name):
    path = Path(path)
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


meth = load(METHOD, 'meth_R25_rankedtrim')
src = load(SRC, 'src_R25_rankedtrim')
res = meth.run_method()
clbt = res[0]
extra = res[4]
clbt_truth = src.get_default_clbt()
dKg_truth = clbt_truth['Kg'] - np.eye(3)
dKa_truth = clbt_truth['Ka'] - np.eye(3)
params = [
    ('eb_x',   clbt_truth['eb'][0] / src.glv.dph,  lambda c: (-c['eb'][0]) / src.glv.dph), ('eb_y', clbt_truth['eb'][1] / src.glv.dph, lambda c: (-c['eb'][1]) / src.glv.dph), ('eb_z', clbt_truth['eb'][2] / src.glv.dph, lambda c: (-c['eb'][2]) / src.glv.dph),
    ('db_x',   clbt_truth['db'][0] / src.glv.ug,   lambda c: (-c['db'][0]) / src.glv.ug), ('db_y', clbt_truth['db'][1] / src.glv.ug, lambda c: (-c['db'][1]) / src.glv.ug), ('db_z', clbt_truth['db'][2] / src.glv.ug, lambda c: (-c['db'][2]) / src.glv.ug),
    ('dKg_xx', dKg_truth[0,0] / src.glv.ppm,       lambda c: (-(c['Kg']-np.eye(3))[0,0]) / src.glv.ppm), ('dKg_yx', dKg_truth[1,0] / src.glv.sec, lambda c: (-(c['Kg']-np.eye(3))[1,0]) / src.glv.sec), ('dKg_zx', dKg_truth[2,0] / src.glv.sec, lambda c: (-(c['Kg']-np.eye(3))[2,0]) / src.glv.sec),
    ('dKg_xy', dKg_truth[0,1] / src.glv.sec,       lambda c: (-(c['Kg']-np.eye(3))[0,1]) / src.glv.sec), ('dKg_yy', dKg_truth[1,1] / src.glv.ppm, lambda c: (-(c['Kg']-np.eye(3))[1,1]) / src.glv.ppm), ('dKg_zy', dKg_truth[2,1] / src.glv.sec, lambda c: (-(c['Kg']-np.eye(3))[2,1]) / src.glv.sec),
    ('dKg_xz', dKg_truth[0,2] / src.glv.sec,       lambda c: (-(c['Kg']-np.eye(3))[0,2]) / src.glv.sec), ('dKg_yz', dKg_truth[1,2] / src.glv.sec, lambda c: (-(c['Kg']-np.eye(3))[1,2]) / src.glv.sec), ('dKg_zz', dKg_truth[2,2] / src.glv.ppm, lambda c: (-(c['Kg']-np.eye(3))[2,2]) / src.glv.ppm),
    ('dKa_xx', dKa_truth[0,0] / src.glv.ppm,       lambda c: (-(c['Ka']-np.eye(3))[0,0]) / src.glv.ppm), ('dKa_xy', dKa_truth[0,1] / src.glv.sec, lambda c: (-(c['Ka']-np.eye(3))[0,1]) / src.glv.sec), ('dKa_xz', dKa_truth[0,2] / src.glv.sec, lambda c: (-(c['Ka']-np.eye(3))[0,2]) / src.glv.sec),
    ('dKa_yy', dKa_truth[1,1] / src.glv.ppm,       lambda c: (-(c['Ka']-np.eye(3))[1,1]) / src.glv.ppm), ('dKa_yz', dKa_truth[1,2] / src.glv.sec, lambda c: (-(c['Ka']-np.eye(3))[1,2]) / src.glv.sec), ('dKa_zz', dKa_truth[2,2] / src.glv.ppm, lambda c: (-(c['Ka']-np.eye(3))[2,2]) / src.glv.ppm),
    ('Ka2_x',  clbt_truth['Ka2'][0] / src.glv.ugpg2, lambda c: (-c['Ka2'][0]) / src.glv.ugpg2), ('Ka2_y', clbt_truth['Ka2'][1] / src.glv.ugpg2, lambda c: (-c['Ka2'][1]) / src.glv.ugpg2), ('Ka2_z', clbt_truth['Ka2'][2] / src.glv.ugpg2, lambda c: (-c['Ka2'][2]) / src.glv.ugpg2),
    ('rx_x', clbt_truth['rx'][0], lambda c: -c['rx'][0]), ('rx_y', clbt_truth['rx'][1], lambda c: -c['rx'][1]), ('rx_z', clbt_truth['rx'][2], lambda c: -c['rx'][2]),
    ('ry_x', clbt_truth['ry'][0], lambda c: -c['ry'][0]), ('ry_y', clbt_truth['ry'][1], lambda c: -c['ry'][1]), ('ry_z', clbt_truth['ry'][2], lambda c: -c['ry'][2]),
]
rows = []
pcts = []
for name, truth, fn in params:
    est = float(fn(clbt))
    abs_err = abs(truth - est)
    pct_err = abs_err / abs(truth) * 100.0 if abs(truth) > 1e-15 else None
    rows.append({'param': name, 'truth': float(truth), 'estimate': est, 'abs_error': abs_err, 'pct_error': pct_err})
    if pct_err is not None:
        pcts.append(pct_err)
summary = {
    'overall_mean_pct_error': float(np.mean(pcts)),
    'overall_median_pct_error': float(np.median(pcts)),
    'worst_param_pct_error': float(np.max(pcts)),
}
OUT.write_text(json.dumps({'method': 'R25_markov_42state_gm1_round25_pairgate_rankedtrim', 'summary': summary, 'rows': rows, 'extra': extra}, ensure_ascii=False, indent=2))
print(f'WROTE {OUT}')
