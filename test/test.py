import pytest
import os
os.environ["RUN_PARALLEL_TESTS"] = "1"
pytest.main(["-vv", "design_pytest.py"])