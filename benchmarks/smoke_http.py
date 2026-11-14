"""Launch the real CLI and check HTTP generation, SSE and cancellation.

Run with a model already cached, or allow its initial Hugging Face download.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryFile() as logs:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mini_vllm",
                "--model",
                args.model,
                "--port",
                str(port),
                "--num-blocks",
                "128",
                "--block-size",
                "4",
            ],
            stdout=logs,
            stderr=logs,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
                for _ in range(600):
                    if process.poll() is not None:
                        raise RuntimeError("Server exited during startup")
                    try:
                        if client.get("/ready").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Readiness timeout")
                assert client.get("/health").json() == {"status": "ok"}
                payload = {"prompt": "Hello world", "max_tokens": 4}
                final = client.post("/generate", json=payload)
                final.raise_for_status()
                streamed = client.post("/generate/stream", json=payload)
                streamed.raise_for_status()
                events = [
                    json.loads(line[6:])
                    for line in streamed.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert len(events) == final.json()["generated_tokens"] + 1
                assert "".join(event.get("text", "") for event in events) == final.json()["text"]
                with client.stream(
                    "POST", "/generate/stream", json={"prompt": "Hello world", "max_tokens": 128}
                ) as response:
                    response.raise_for_status()
                    key = response.headers["x-request-id"]
                    cancelled = client.post(f"/requests/{key}/cancel")
                    cancelled.raise_for_status()
                    assert cancelled.json()["cancelled"]
                    body = response.read().decode()
                    assert '"finish_reason": "cancelled"' in body
                assert client.get("/ready").status_code == 200
                print(
                    json.dumps(
                        {
                            "http_smoke": "passed",
                            "model": args.model,
                            "generated_tokens": final.json()["generated_tokens"],
                            "stream_reconstruction": True,
                            "cancellation": True,
                        }
                    )
                )
        except Exception:
            logs.seek(0)
            print(logs.read().decode(errors="replace"), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
