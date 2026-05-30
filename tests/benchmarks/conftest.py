"""conftest for HybridDB benchmarks — standalone OSS variant."""

from pathlib import Path

import pytest

from hybriddb import HybridDB

from .helpers import FULL, SMOKE, Scale, archive_results


def pytest_addoption(parser):
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="Run benchmark tests (skipped by default during normal pytest)",
    )
    parser.addoption(
        "--benchmark-full",
        action="store_true",
        default=False,
        help="Run full-scale benchmarks (default: smoke)",
    )
    parser.addoption(
        "--precompute-embeddings",
        action="store_true",
        default=True,
        help="Pre-compute and cache embeddings (default: true)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-benchmarks") or config.getoption("--benchmark-full"):
        return
    skip_benchmark = pytest.mark.skip(reason="benchmark tests require --run-benchmarks")
    benchmark_dir = Path(__file__).parent
    for item in items:
        if Path(str(item.path)).is_relative_to(benchmark_dir):
            item.add_marker(skip_benchmark)


@pytest.fixture(scope="session")
def embedding_fn():
    """Session-scoped SentenceTransformer model — loaded once."""
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda text: model.encode(text).tolist()


@pytest.fixture
def scale(request) -> Scale:
    return FULL if request.config.getoption("--benchmark-full") else SMOKE


@pytest.fixture
def db(request, embedding_fn, tmp_path) -> HybridDB:
    h = HybridDB(
        path=str(tmp_path),
        embedding_fn=embedding_fn,
        embedding_model_name="all-MiniLM-L6-v2",
    )
    yield h
    try:
        h.close()
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    json_path = session.config.getoption("--benchmark-json")
    if json_path:
        path = getattr(json_path, "name", json_path)
        if path and Path(path).exists():
            archive_results(path)
