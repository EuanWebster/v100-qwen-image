"""Research preparation must not alter production or mislabel Q4 evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_isolated_q4_preparation_preserves_product_and_quant_labels(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = root / 'src/qwen_engine/config.py'
    before = hashlib.sha256(config.read_bytes()).digest()
    dest = tmp_path / 'experiment'
    subprocess.run([sys.executable, str(root / 'scripts/prepare_q4_experiment.py'), str(dest)], check=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(dest / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    code = '''import json, qwen_engine
from qwen_engine.config import GenerationConfig, Quant
from qwen_engine.q4_artifact import Q4_QUANT_FORMAT
c=GenerationConfig(prompt="cube",quant="q4")
assert c.quant is Quant.Q4 and Quant.Q8.value=="q8"
assert "q4_manifest_path" in c.to_dict()
assert Q4_QUANT_FORMAT=="gguf-q4_k_m"
print(json.dumps({"file":qwen_engine.__file__,"quant":c.to_dict()["quant"]}))'''
    result = subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)
    got = json.loads(result.stdout)
    assert Path(got['file']).is_relative_to(dest)
    assert got['quant'] == 'q4'
    assert hashlib.sha256(config.read_bytes()).digest() == before
    assert 'q4 is not implemented' in (root / 'src/qwen_engine/adapters/qwen21_diffusers.py').read_text()
    manifest = json.loads((root / 'models/q4_eval_manifest.json').read_text())
    assert manifest['quant_format'] == 'gguf-q4_k_m'
    assert manifest['bytes'] == 4199565024
