"""One catalog drives CLI and generated browser presets without ML imports."""
import re
from pathlib import Path

import pytest
import qwen_engine
from qwen_engine import cli
from qwen_engine.presets import PRESETS, browser_catalog, preset_config


def test_canonical_upstream_sizes_and_aligned_companions():
    assert [(p['width'], p['height']) for p in PRESETS.values() if p['tier'] == '2K'] == [
        (2048, 2048), (2400, 1792), (1792, 2400), (2528, 1696),
        (1696, 2528), (2752, 1536), (1536, 2752)]
    assert len(PRESETS) == 14
    assert all(p['width'] % 32 == p['height'] % 32 == 0 for p in PRESETS.values())
    assert {(p['ratio'], p['tier']) for p in PRESETS.values()} == {
        (ratio, tier) for ratio in ('1:1','4:3','3:4','3:2','2:3','16:9','9:16') for tier in ('1K','2K')}


def test_packaged_browser_catalog_matches_shared_source():
    source = (Path(qwen_engine.__file__).parent / 'web_static/app.js').read_text()
    block = re.search(r'  // BEGIN GENERATED PRESETS.*?  // END GENERATED PRESETS', source, re.S)
    assert block and block.group() == browser_catalog(), 'run scripts/sync_presets.py'


@pytest.mark.parametrize('name', PRESETS)
def test_cli_preset_dimensions_and_decode_policy(name):
    args = cli.build_parser().parse_args(['--prompt', 'cube', '--preset', name])
    config = cli._load_config(args)
    assert config.width == PRESETS[name]['width']
    assert config.height == PRESETS[name]['height']
    assert config.vae_tiled_decode == name.endswith('2k')


def test_explicit_flags_override_preset():
    args = cli.build_parser().parse_args(['--prompt', 'cube', '--preset', 'landscape2k',
        '--width', '640', '--height', '480', '--no-tiled-decode'])
    config = cli._load_config(args)
    assert (config.width, config.height, config.vae_tiled_decode) == (640, 480, False)
