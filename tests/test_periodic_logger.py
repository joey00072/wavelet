import logging
import threading

from wavelet.orchestrator.periodic_logger import PeriodicLogger, pipeline_status


def test_pipeline_status_is_one_line() -> None:
    status = pipeline_status(
        policy_step=4,
        published=8,
        target=20,
        submitted=11,
        inflight=3,
        ready=1,
        policy_loading=True,
    )

    assert status == (
        "Pipeline | policy=4 | published=8/20 | submitted=11 | inflight=3 | "
        "ready=1 | policy_load=pending"
    )
    assert "\n" not in status


def test_periodic_logger_emits_while_running(caplog) -> None:
    emitted = threading.Event()

    def collect() -> str:
        emitted.set()
        return "Pipeline | healthy"

    with (
        caplog.at_level(logging.INFO),
        PeriodicLogger(collect, interval_seconds=0.01),
    ):
        assert emitted.wait(timeout=1.0)

    assert "Pipeline | healthy" in caplog.text
