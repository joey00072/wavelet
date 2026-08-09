from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import torch

from wavelet.configs.rl_config import OPDAlgorithmConfig, RLDataConfig, RLLossConfig
from wavelet.data.rl import (
    RLExample,
    collate_rl_batch,
    component_loss_counts,
    prepare_rl_sample,
)
from wavelet.orchestrator.algorithms import OPDAlgorithm, score_algorithm_records
from wavelet.trainer.losses import compute_loss


def test_opd_teacher_scores_flow_through_collation_and_loss() -> None:
    requests: list[dict[str, object]] = []

    class TeacherHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            size = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(size))
            requests.append({"path": self.path, "payload": payload})
            token_ids = payload["token_ids"]
            prompt_logprobs = [None]
            prompt_logprobs.extend(
                {str(token_id): {"logprob": -0.1 * index}}
                for index, token_id in enumerate(token_ids[1:], start=1)
            )
            response = json.dumps({"prompt_logprobs": prompt_logprobs}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), TeacherHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        algorithm = OPDAlgorithm(
            OPDAlgorithmConfig(
                teacher={
                    "name": "teacher",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                }
            )
        )
        record = RLExample(
            prompt=[],
            completion=[],
            advantage=None,
            reward=None,
            input_ids=[1, 2],
            target_ids=[2, 3],
            loss_mask=[False, True],
            inference_logprobs=[-0.4],
            temperatures=1.0,
            source="opd",
        )

        scored = score_algorithm_records(algorithm, [record], scope="rollout")[0]
        sample = prepare_rl_sample(
            scored,
            tokenizer=None,  # type: ignore[arg-type]
            data_config=RLDataConfig(seq_len=8),
            seq_len=8,
        )
        assert sample is not None
        batch = collate_rl_batch([sample], pad_token_id=0)
        trainer_logprobs = torch.tensor(
            [[-0.3, -0.25]],
            dtype=torch.float32,
            requires_grad=True,
        )
        output = compute_loss(
            trainer_logprobs,
            batch["inference_logprobs"],
            None,
            batch["advantages"],
            batch["loss_mask"],
            RLLossConfig(),
            position_ids=batch["position_ids"],
            ref_logprobs=batch["ref_logprobs"],
            rl_weights=batch["rl_weights"],
            ce_weights=batch["ce_weights"],
            ref_kl_weights=batch["ref_kl_weights"],
            component_loss_scales=component_loss_counts(sample),
        )
        output.loss.backward()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert requests == [
        {
            "path": "/inference/v1/generate",
            "payload": {
                "model": "teacher",
                "token_ids": [1, 2, 3],
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "prompt_logprobs": 1,
                },
            },
        }
    ]
    assert scored.ref_logprobs == pytest.approx([-0.2])
    assert torch.isfinite(output.loss)
    assert trainer_logprobs.grad is not None
    assert trainer_logprobs.grad[0, 0].item() == 0.0
    assert trainer_logprobs.grad[0, 1].item() != 0.0
