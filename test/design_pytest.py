import os
import sys
from pathlib import Path
import multiprocessing as mp
import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
print("ROOT:", ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Quiet logs during tests
os.environ.setdefault("ASTROPY_LOG_LEVEL", "ERROR")

@pytest.fixture(scope="session", autouse=True)
def set_spawn_on_macos():
    # Make behavior match macOS defaults and your examples
    if sys.platform == "darwin":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

@pytest.fixture(scope="session")
def example_input():
    # Prefer example file if present
    p = ROOT / "examples" / "input_template.dustpol"
    if not p.exists():
        pytest.skip("examples/input_template.dustpol not found")
    return str(p)

@pytest.fixture(scope="session")
def example_input_starless():
    # Prefer example file if present
    p = ROOT / "examples" / "input_template_starless.dustpol"
    if not p.exists():
        pytest.skip("examples/input_template_starless.dustpol not found")
    return str(p)

@pytest.fixture(scope="session")
def example_input_protostar():
    p = ROOT / "examples" / "input_template_protostar.dustpol"
    if not p.exists():
        pytest.skip("examples/input_template_protostar.dustpol not found")
    return str(p)

import numpy as np
from DustPOL_py import DustPOL

def test_extinction_curve_finite(example_input_starless):
    exe = DustPOL(example_input_starless, parallel=False, nsample=16)
    w, A_per_Ngas = exe.extinction_curve(verbose=False)
    assert w.ndim == 1 and A_per_Ngas.shape == w.shape
    assert np.isfinite(w).all() and np.isfinite(A_per_Ngas).all()
    assert (w > 0).all()
    assert (A_per_Ngas >= 0).all()
    
def test_override_gsd_law_hd23(example_input_starless):
    exe = DustPOL(example_input_starless, GSD_law="HD23", parallel=False, nsample=8)
    assert exe.GSD_law.lower() == "hd23"

def test_override_f_min_dg_to_none(example_input_starless):
    exe = DustPOL(example_input_starless, f_min="DG", parallel=False)
    assert exe.f_min is None

def test_override_output_dir_normalized(tmp_path, example_input_starless):
    out = tmp_path / "run1"
    exe = DustPOL(example_input_starless, output_dir=str(out), parallel=False)
    # ensure directory got created and normalized
    assert Path(exe.output_dir).exists()
    assert Path(exe.output_dir).resolve() == out.resolve()
    

@pytest.mark.skipif(
    os.environ.get("RUN_PARALLEL_TESTS", "0") != "1",
    reason="Set RUN_PARALLEL_TESTS=1 to run parallel tests",
)
def test_cal_pol_emi_small(example_input, tmp_path):
    exe = DustPOL(example_input, output_dir=str(tmp_path))
    exe.cal_pol_emi(Av=0.0, verbose=False, save_output=True, filename_output="smoke")
    assert (tmp_path / "smoke_emi.dat").exists()

def test_cal_pol_abs_small(example_input_starless, tmp_path):
    exe = DustPOL(example_input_starless, output_dir=str(tmp_path))
    # request writing to disk; default file is p_abs.dat in output_dir
    exe.cal_pol_abs(NH=0.0, verbose=False, save_output=True, filename_output="smoke")
    assert (tmp_path / "smoke_abs.dat").exists()
    
def test_isoCloud_pos_small(example_input_starless, tmp_path):
    exe = DustPOL(example_input_starless, parallel=False, nsample=8, output_dir=str(tmp_path))
    exe.parallel = True
    exe.max_workers = 2
    exe.isoCloud_pos(filename_output="smoke", progress=False)
    assert (tmp_path / "smoke_abs.dat").exists()
    assert (tmp_path / "smoke_emi.dat").exists()
    
def test_isoProtostar_pos_small(example_input_protostar, tmp_path):
    exe = DustPOL(example_input_protostar, parallel=False, nsample=8, output_dir=str(tmp_path))
    exe.parallel = True
    exe.max_workers = 2
    exe.isoProtostar_pos(filename_output="smoke", progress=False)
    assert (tmp_path / "smoke_abs.dat").exists()
    assert (tmp_path / "smoke_emi.dat").exists()