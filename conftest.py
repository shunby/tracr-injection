import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )

def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("Skipping slow test: use --run-slow to run it")

