# -*- coding: utf-8 -*-
"""Minimal test runner (pytest is not required):  python tests/run_tests.py
Writes the report to results/tests_report.txt and exits with status 1 if any test fails."""
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    lines, failed = [], 0
    for fn in sorted(f for f in os.listdir(HERE) if f.startswith('test_') and f.endswith('.py')):
        spec = importlib.util.spec_from_file_location(fn[:-3], os.path.join(HERE, fn))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith('test_')):
            t0 = time.perf_counter()
            try:
                getattr(mod, name)()
                status = 'PASS'
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == 'Skip':
                    status = f'SKIP ({e})'
                else:
                    status = 'FAIL'
                    failed += 1
                    lines.append(traceback.format_exc())
            lines.append(f'{status:6s} {fn}::{name}  ({time.perf_counter() - t0:.1f}s)')
            print(lines[-1], flush=True)
    summary = f'{"ALL TESTS PASSED" if failed == 0 else f"{failed} TEST(S) FAILED"}'
    lines.append(summary)
    print(summary)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'tests_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return failed


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
