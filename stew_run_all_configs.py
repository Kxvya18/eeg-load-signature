from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple
PROJECT_ROOT = Path(__file__).resolve().parent

def discover_suffixes() -> List[str]:
    suffixes = []
    for p in PROJECT_ROOT.glob('stew_tda_results_final_*'):
        if p.is_dir():
            suffixes.append(p.name.replace('stew_tda_results_final_', '', 1))
    return sorted(suffixes)

def check_required_inputs(suffix: str) -> Tuple[bool, List[str]]:
    in_dir = PROJECT_ROOT / f'stew_tda_results_final_{suffix}'
    required = [in_dir / 'subject_level_features.csv', in_dir / 'preprocessing_log.csv', in_dir / 'pipeline_config.json', in_dir / 'per_subject_raw']
    missing = [str(p) for p in required if not p.exists()]
    return (len(missing) == 0, missing)

def run_step(cmd: List[str], log_path: Path, dry_run: bool) -> bool:
    print(f"    $ {' '.join(cmd)}")
    if dry_run:
        return True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, 'w') as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
    dt = time.time() - t0
    ok = result.returncode == 0
    status = 'OK' if ok else f'FAILED (exit {result.returncode})'
    print(f'      -> {status} in {dt:.1f}s  (log: {log_path})')
    if not ok:
        tail = log_path.read_text().splitlines()[-15:]
        print('      ---- last lines of log ----')
        for line in tail:
            print(f'      {line}')
        print('      ----------------------------')
    return ok

def run_one_config(suffix: str, python_exe: str, dry_run: bool, include_pdrh: bool=False, skip_nwcf: bool=False) -> dict:
    print(f"\n{'=' * 78}\nCONFIG: {suffix}\n{'=' * 78}")
    (ok_inputs, missing) = check_required_inputs(suffix)
    if not ok_inputs:
        print(f'  SKIPPED — missing required input(s):')
        for m in missing:
            print(f'    - {m}')
        return {'suffix': suffix, 'status': 'skipped_missing_inputs'}
    in_dir = PROJECT_ROOT / f'stew_tda_results_final_{suffix}'
    out_dir = PROJECT_ROOT / f'stew_output_{suffix}'
    fig_dir = PROJECT_ROOT / f'manuscript_figures_{suffix}'
    log_dir = PROJECT_ROOT / 'stew_run_all_configs_logs' / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {'suffix': suffix, 'status': 'ok'}
    print('  [1/3] stew_advanced_analysis.py  (Tables 1-16, Figures 1-5, advanced_topology_features.csv — now via the v2 NWCF filtration)')
    cmd1 = [python_exe, 'stew_advanced_analysis.py', '--input-dir', str(in_dir), '--output-dir', str(out_dir)]
    if include_pdrh:
        cmd1.append('--include-pdrh')
    if skip_nwcf:
        cmd1.append('--skip-nwcf')
    ok1 = run_step(cmd1, log_dir / '01_advanced_analysis.log', dry_run)
    result['advanced_analysis'] = 'ok' if ok1 else 'failed'
    print('  [2/3] stew_methodological_justification.py  (Goal 1/2/3 narrative report)')
    report_path = out_dir / 'methodological_justification_report.txt'
    cmd2 = [python_exe, 'stew_methodological_justification.py', '--suffix', suffix, '--output-dir', str(out_dir)]
    if include_pdrh:
        cmd2.append('--include-pdrh')
    print(f"    $ {' '.join(cmd2)}  >  {report_path}")
    ok2 = True
    if not dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            r = subprocess.run(cmd2, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        ok2 = r.returncode == 0
        print(f"      -> {('OK' if ok2 else f'FAILED (exit {r.returncode})')}  (report: {report_path})")
    result['methodological_justification'] = 'ok' if ok2 else 'failed'
    print('  [3/3] stew_manuscript_figures.py  (Figures 1-4, including the rebuilt NWCF Figure 2)')
    cmd3 = [python_exe, 'stew_manuscript_figures.py', '--input_dir', str(in_dir), '--output_dir', str(fig_dir)]
    if include_pdrh:
        cmd3.append('--include-pdrh')
    ok3 = run_step(cmd3, log_dir / '03_manuscript_figures.log', dry_run)
    result['manuscript_figures'] = 'ok' if ok3 else 'failed'
    if not (ok1 and ok2 and ok3):
        result['status'] = 'completed_with_errors'
    return result

def run_robustness_summary(succeeded_suffixes: List[str], python_exe: str, dry_run: bool) -> bool:
    if not succeeded_suffixes:
        print('\nNo configs completed successfully — skipping cross-config robustness summary.')
        return False
    out_dirs = [str(PROJECT_ROOT / f'stew_output_{s}') for s in succeeded_suffixes]
    print(f"\n{'=' * 78}\nCROSS-CONFIG ROBUSTNESS SUMMARY  ({len(out_dirs)} configs)\n{'=' * 78}")
    log_path = PROJECT_ROOT / 'stew_run_all_configs_logs' / 'robustness_summary.log'
    return run_step([python_exe, 'stew_robustness.py', *out_dirs, '--output-dir', 'stew_robustness_summary'], log_path, dry_run)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suffixes', nargs='+', default=None, help='Explicit list of config suffixes, e.g. 123plv 83aec 84plv')
    ap.add_argument('--auto-discover', action='store_true', help='Glob for every stew_tda_results_final_* directory instead of passing --suffixes explicitly.')
    ap.add_argument('--python', default=sys.executable, help='Python executable to use for the subprocess calls (defaults to the interpreter running this script).')
    ap.add_argument('--dry-run', action='store_true', help='Print every command that would run, without running anything.')
    ap.add_argument('--include-pdrh', action='store_true', help='Opt into PDRH outputs. Default is NWCF-only.')
    ap.add_argument('--skip-nwcf', action='store_true', help='Forwarded to stew_advanced_analysis.py — skip NWCF entirely. Provided for symmetry; not the current use case.')
    args = ap.parse_args()
    if args.auto_discover:
        suffixes = discover_suffixes()
        if not suffixes:
            print(f'No stew_tda_results_final_* directories found in {PROJECT_ROOT}. Nothing to do.')
            return
        print(f'Auto-discovered {len(suffixes)} config(s): {suffixes}')
    elif args.suffixes:
        suffixes = args.suffixes
    else:
        ap.error('Pass either --suffixes <list> or --auto-discover.')
        return
    results = [run_one_config(s, args.python, args.dry_run, include_pdrh=args.include_pdrh, skip_nwcf=args.skip_nwcf) for s in suffixes]
    succeeded = [r['suffix'] for r in results if r['status'] in ('ok', 'completed_with_errors')]
    run_robustness_summary(succeeded, args.python, args.dry_run)
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"{'suffix':12s} {'status':24s} {'advanced':10s} {'justif.':10s} {'figures':10s}")
    for r in results:
        print(f"{r['suffix']:12s} {r['status']:24s} {r.get('advanced_analysis', '-'):10s} {r.get('methodological_justification', '-'):10s} {r.get('manuscript_figures', '-'):10s}")
    n_ok = sum((1 for r in results if r['status'] == 'ok'))
    print(f'\n{n_ok}/{len(results)} configs completed cleanly. Per-step logs are under stew_run_all_configs_logs/.')
if __name__ == '__main__':
    main()
