"""Prepare an isolated Q4_K_M evaluation package; never change the live loader.

This deliberately derives the narrow experiment from the reviewed Q8 path.
The copy has real q4 config/calibration labels and separate artifact identities.
It retains the hardware, manifest, source identity, cancellation and encoder
gates. Q4 stays refused by the product until its own integration is reviewed.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'src/qwen_engine'
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'src/qwen_engine'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    originals = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*.py')}
    for filename in ('config.py', 'cli.py', 'calibration.py',
                     'adapters/qwen21_diffusers.py', 'adapters/supplied_qwen21.py'):
        path = target / filename
        text = path.read_text()
        assert 'q8' in text, filename
        if filename == 'config.py':
            text = text.replace('q8_manifest', 'q4_manifest')
        else:
            text = text.replace('q8', 'q4').replace('Q8', 'Q4')
        path.write_text(text)
    artifact = (source / 'q8_artifact.py').read_text().replace('q8', 'q4').replace('Q8', 'Q4')
    artifact = artifact.replace('gguf-q4_0', 'gguf-q4_k_m')
    (target / 'q4_artifact.py').write_text(artifact)
    loader = (source / 'adapters/q8_loader.py').read_text().replace('q8', 'q4').replace('Q8', 'Q4')
    loader = loader.replace('Q4_0', 'Q4_K')
    loader = loader.replace('gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.F16',
                            'gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K, gguf.GGMLQuantizationType.Q6_K, gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.F16')
    loader = loader.replace('parameter.quant_type != gguf.GGMLQuantizationType.Q4_K',
                            'parameter.quant_type not in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K, gguf.GGMLQuantizationType.Q6_K, gguf.GGMLQuantizationType.Q8_0)')
    (target / 'adapters/q4_loader.py').write_text(loader)
    changes = []
    for path in sorted(target.rglob('*.py')):
        name = str(path.relative_to(target))
        after = path.read_bytes()
        before = originals.get(name, b'')
        if after != before:
            changes.extend(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                              fromfile='base/' + name, tofile='experiment/' + name))
    (destination / 'experiment.diff').write_text(''.join(changes))
    metadata = {
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'base_python_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in originals.items()},
        'scope': 'isolated Q4_K_M evaluation only; FP16 computation; Q4_K/Q5_K/Q6_K/Q8_0 mixed tensors',
    }
    (destination / 'provenance.json').write_text(json.dumps(metadata, indent=2))
    print(destination)


if __name__ == '__main__':
    main()
