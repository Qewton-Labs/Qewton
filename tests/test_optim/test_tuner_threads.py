"""The tuner limits the CPU threads of its worker processes, so several
workers do not compete for the same CPUs."""

import multiprocessing as mp
import os
from types import SimpleNamespace

import pytest

from qewton.optim.tuner import base as tuner_base
from qewton.optim.tuner.base import THREAD_ENV_VARS, Tuner, thread_limit


def _report_threads(queue):
    import torch  # pylint: disable=import-outside-toplevel

    queue.put((torch.get_num_threads(), os.environ.get("OMP_NUM_THREADS")))


@pytest.fixture
def clean_env(monkeypatch):
    for name in THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_spawned_process_uses_the_limit(clean_env):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with thread_limit(3):
        process = ctx.Process(target=_report_threads, args=(queue,))
        process.start()
    threads, env_value = queue.get(timeout=120)
    process.join()

    assert threads == 3
    assert env_value == "3"


def test_limit_is_restored(clean_env, monkeypatch):
    monkeypatch.setenv("MKL_NUM_THREADS", "7")
    with thread_limit(2):
        assert all(os.environ[name] == "2" for name in THREAD_ENV_VARS)
    assert "OMP_NUM_THREADS" not in os.environ
    assert os.environ["MKL_NUM_THREADS"] == "7"


def test_none_keeps_the_environment(clean_env):
    with thread_limit(None):
        assert not any(name in os.environ for name in THREAD_ENV_VARS)


def _tuner(threads_per_worker, process_number):
    # worker_threads only reads these two attributes.
    return SimpleNamespace(
        threads_per_worker=threads_per_worker, process_number=process_number
    )


def test_cpus_are_split_between_workers(clean_env, monkeypatch):
    monkeypatch.setattr(tuner_base, "available_cpus", lambda: 56)
    assert Tuner.worker_threads(_tuner(None, 8)) == 7
    assert Tuner.worker_threads(_tuner(None, 100)) == 1


def test_explicit_threads_win(clean_env, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert Tuner.worker_threads(_tuner(2, 8)) == 2


def test_set_environment_is_kept(clean_env, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert Tuner.worker_threads(_tuner(None, 8)) is None
