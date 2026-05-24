#!/usr/bin/env python3
"""Export installed package versions for Python imports used in this repo.

Run on the target server environment:
  python scripts/export_requirements_from_imports.py

Outputs by default:
  requirements_detected.txt
  requirements_detected_report.txt
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
import sysconfig
from importlib import metadata
from pathlib import Path


IGNORE_DIRS = {
    '.git', '.hg', '.svn', '.mypy_cache', '.pytest_cache', '.ruff_cache',
    '.pycache', '__pycache__', '.venv', 'venv', 'env', 'node_modules',
    'saved_models', 'checkpoints', 'logs', 'datasets',
}

FALLBACK_IMPORT_TO_DIST = {
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'torchaudio': 'torchaudio',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'tqdm': 'tqdm',
    'einops': 'einops',
    'lightning': 'lightning',
    'pytorch_lightning': 'pytorch-lightning',
    'torchmetrics': 'torchmetrics',
    'omegaconf': 'omegaconf',
    'hydra': 'hydra-core',
    'transformers': 'transformers',
    'datasets': 'datasets',
    'tensorboard': 'tensorboard',
    'seaborn': 'seaborn',
    'statsmodels': 'statsmodels',
    'patoolib': 'patool',
}


def iter_py_files(root: Path):
    for path in root.rglob('*.py'):
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue
        yield path


def collect_imports(root: Path):
    imports: dict[str, set[str]] = {}
    parse_errors: list[tuple[str, str]] = []
    for path in iter_py_files(root):
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except Exception as exc:  # noqa: BLE001
            parse_errors.append((str(path.relative_to(root)), repr(exc)))
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    names = [node.module.split('.')[0]]
            for name in names:
                imports.setdefault(name, set()).add(str(path.relative_to(root)))
    return imports, parse_errors


def local_top_level_names(root: Path) -> set[str]:
    names = set()
    for child in root.iterdir():
        if child.name.startswith('.'):
            continue
        if child.is_dir() and (child / '__init__.py').exists():
            names.add(child.name)
        elif child.is_file() and child.suffix == '.py':
            names.add(child.stem)
    return names


def stdlib_names() -> set[str]:
    names = set(getattr(sys, 'stdlib_module_names', set()))
    stdlib_path = Path(sysconfig.get_paths().get('stdlib', ''))
    if stdlib_path.exists():
        for child in stdlib_path.iterdir():
            if child.name.startswith('_'):
                continue
            if child.is_dir() and (child / '__init__.py').exists():
                names.add(child.name)
            elif child.suffix == '.py':
                names.add(child.stem)
    return names


def is_probably_local(root: Path, module: str, local_names: set[str]) -> bool:
    if module in local_names:
        return True
    if (root / f'{module}.py').exists():
        return True
    if (root / module).is_dir():
        return True
    return False


def build_distribution_map():
    mapping = metadata.packages_distributions()
    normalized = {k: v for k, v in mapping.items()}
    return normalized


def choose_distribution(import_name: str, packages_to_dists: dict[str, list[str]]) -> str | None:
    if import_name in FALLBACK_IMPORT_TO_DIST:
        return FALLBACK_IMPORT_TO_DIST[import_name]
    dists = packages_to_dists.get(import_name)
    if dists:
        return sorted(dists, key=lambda x: (len(x), x.lower()))[0]
    return None


def module_origin(import_name: str) -> str:
    try:
        spec = importlib.util.find_spec(import_name)
    except Exception as exc:  # noqa: BLE001
        return f'find_spec_error: {exc!r}'
    if spec is None:
        return 'not importable'
    return str(spec.origin or 'namespace/package')


def main():
    parser = argparse.ArgumentParser(description='Export requirements from repo imports using installed server versions.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Repository root to scan')
    parser.add_argument('--output', type=Path, default=Path('requirements_detected.txt'),
                        help='Output requirements file')
    parser.add_argument('--report', type=Path, default=Path('requirements_detected_report.txt'),
                        help='Output detailed report')
    parser.add_argument('--include-local', action='store_true',
                        help='Do not filter local top-level modules')
    args = parser.parse_args()

    root = args.root.resolve()
    imports, parse_errors = collect_imports(root)
    local_names = local_top_level_names(root)
    stdlib = stdlib_names()
    packages_to_dists = build_distribution_map()

    requirements: dict[str, str] = {}
    skipped: dict[str, str] = {}
    unresolved: dict[str, str] = {}

    for import_name in sorted(imports):
        if import_name in stdlib:
            skipped[import_name] = 'stdlib'
            continue
        if not args.include_local and is_probably_local(root, import_name, local_names):
            skipped[import_name] = 'local'
            continue
        dist = choose_distribution(import_name, packages_to_dists)
        if dist is None:
            unresolved[import_name] = module_origin(import_name)
            continue
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            unresolved[import_name] = f'distribution not installed: {dist}; origin={module_origin(import_name)}'
            continue
        requirements[dist] = version

    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    with output.open('w', encoding='utf-8') as f:
        f.write('# Generated by scripts/export_requirements_from_imports.py\n')
        f.write(f'# Python: {sys.version.split()[0]}\n')
        f.write(f'# Repo: {root}\n')
        for dist in sorted(requirements, key=str.lower):
            f.write(f'{dist}=={requirements[dist]}\n')

    with report.open('w', encoding='utf-8') as f:
        f.write(f'Root: {root}\n')
        f.write(f'Python: {sys.version}\n\n')
        f.write('Resolved requirements:\n')
        for dist in sorted(requirements, key=str.lower):
            used_by = sorted({file for imp, files in imports.items() if choose_distribution(imp, packages_to_dists) == dist for file in files})
            f.write(f'  {dist}=={requirements[dist]}\n')
            for file in used_by[:20]:
                f.write(f'    - {file}\n')
            if len(used_by) > 20:
                f.write(f'    ... {len(used_by) - 20} more files\n')
        f.write('\nUnresolved imports:\n')
        for imp, reason in sorted(unresolved.items()):
            f.write(f'  {imp}: {reason}\n')
            for file in sorted(imports[imp])[:10]:
                f.write(f'    - {file}\n')
        f.write('\nSkipped imports:\n')
        for imp, reason in sorted(skipped.items()):
            f.write(f'  {imp}: {reason}\n')
        f.write('\nParse errors:\n')
        for file, error in parse_errors:
            f.write(f'  {file}: {error}\n')

    print(f'Wrote requirements: {output}')
    print(f'Wrote report      : {report}')
    print(f'Resolved packages : {len(requirements)}')
    print(f'Unresolved imports: {len(unresolved)}')
    if unresolved:
        print('Check unresolved imports in the report before using the requirements file.')


if __name__ == '__main__':
    main()
