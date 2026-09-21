from pathlib import Path


def pytest_addoption(parser):
    group = parser.getgroup("prdsim validation")
    group.addoption(
        "--prdsim-validation-report",
        type=Path,
        default=None,
        help="write the structured Gate 3 validation record to this JSON path",
    )
