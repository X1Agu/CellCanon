"""Check dependency versions and CUDA availability."""
from importlib import metadata
from pathlib import Path
import argparse
import json
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-transitive', action='store_true',
                        help='Also check the recorded runtime dependency snapshot.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / ('environments/cuda-py311.constraints.txt' if args.include_transitive else 'requirements.txt')
    checks = {}
    issues = []
    for line in source.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==', 1)
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        matches = actual is not None and actual.split('+', 1)[0] == expected.split('+', 1)[0]
        checks[name] = {'expected': expected, 'installed': actual, 'matches': matches}
        if not matches:
            issues.append({'component': name, 'reason': 'missing_or_different_version'})
    python = platform.python_version()
    if python != '3.11.15':
        issues.append({'component': 'python', 'reason': 'reference_version_is_3.11.15'})
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'AMD64'}:
        issues.append({'component': 'platform', 'reason': 'validated_target_is_linux_x86_64'})
    cuda = {'expected_build': '13.0', 'installed_build': None, 'available': False}
    try:
        import torch
        cuda.update(installed_build=torch.version.cuda, available=torch.cuda.is_available())
        if cuda['installed_build'] != '13.0' or not cuda['available']:
            issues.append({'component': 'cuda', 'reason': 'reference_cuda_13_gpu_unavailable'})
    except Exception as exc:
        issues.append({'component': 'torch', 'reason': 'import_failed', 'error_type': type(exc).__name__})
    print(json.dumps({'status': 'PASS' if not issues else 'MISMATCH', 'python': python,
                      'cuda': cuda, 'libraries': checks, 'issues': issues,
                      'scope': 'Dependency and CUDA availability check.'}, indent=2))
    return 0 if not issues else 2


if __name__ == '__main__':
    sys.exit(main())
