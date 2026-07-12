from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = {'123plv': dict(window_seconds=3.0, surrogates=12, fc_metric='alpha_plv'), '254aec': dict(window_seconds=4.0, surrogates=25, fc_metric='alpha_envelope_correlation'), '83aec': dict(window_seconds=3.0, surrogates=8, fc_metric='alpha_envelope_correlation'), '84aec': dict(window_seconds=4.0, surrogates=8, fc_metric='alpha_envelope_correlation'), '84corr': dict(window_seconds=4.0, surrogates=8, fc_metric='alpha_correlation'), '84plv': dict(window_seconds=4.0, surrogates=8, fc_metric='alpha_plv')}

def load_existing_config(suffix: str) -> dict:
    cfg_path = PROJECT_ROOT / f'stew_tda_results_final_{suffix}' / 'pipeline_config.json'
    if not cfg_path.exists():
        return DEFAULT_CONFIGS[suffix]
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    return dict(window_seconds=float(cfg.get('window_seconds', DEFAULT_CONFIGS[suffix]['window_seconds'])), surrogates=int(cfg.get('surrogate_n_per_method', cfg.get('surrogate_n', DEFAULT_CONFIGS[suffix]['surrogates']))), fc_metric=str(cfg.get('fc_connectivity_metric', DEFAULT_CONFIGS[suffix]['fc_metric'])))

def run(cmd: list[str], dry_run: bool) -> None:
    print('$ ' + ' '.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffixes', nargs='+', default=list(DEFAULT_CONFIGS), help='Config suffixes to rerun.')
    parser.add_argument('--stage', choices=['raw', 'advanced', 'all'], default='all', help='raw = stew_pipeline only; advanced = analysis only; all = both.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--normalization', default='unit_diameter')
    parser.add_argument('--iaaft-iterations', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-pdrh', action='store_true', help='Also generate PDRH advanced outputs. Default is NWCF-only.')
    args = parser.parse_args()
    suffixes = args.suffixes
    unknown = [s for s in suffixes if s not in DEFAULT_CONFIGS]
    if unknown:
        raise SystemExit(f'Unknown suffixes: {unknown}. Known: {sorted(DEFAULT_CONFIGS)}')
    if args.stage in ('raw', 'all'):
        for suffix in suffixes:
            cfg = load_existing_config(suffix)
            out_dir = PROJECT_ROOT / f'stew_tda_results_final_{suffix}'
            cmd = [args.python, 'stew_pipeline.py', 'full', '--output-dir', str(out_dir), '--normalization', args.normalization, '--surrogates', str(cfg['surrogates']), '--iaaft-iterations', str(args.iaaft_iterations), '--window-seconds', str(cfg['window_seconds']), '--fc-metric', cfg['fc_metric']]
            print(f'\n=== RAW PIPELINE: {suffix} ===')
            run(cmd, args.dry_run)
    if args.stage in ('advanced', 'all'):
        cmd = [args.python, 'stew_run_all_configs.py', '--suffixes', *suffixes]
        if args.include_pdrh:
            cmd.append('--include-pdrh')
        print('\n=== ADVANCED NWCF ANALYSIS ===')
        run(cmd, args.dry_run)
if __name__ == '__main__':
    main()
