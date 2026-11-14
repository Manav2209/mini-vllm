import argparse
import logging

import uvicorn

from .server import ServerConfig, create_app


def main():
    parser = argparse.ArgumentParser(description="mini-vLLM CPU inference server (one worker)")
    defaults = ServerConfig.from_env()
    parser.add_argument("--model", default=defaults.model)
    for name in (
        "num_blocks",
        "block_size",
        "max_batch_tokens",
        "max_active_requests",
        "max_queue_size",
        "max_new_tokens",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level", default="info", choices=("debug", "info", "warning", "error")
    )
    args = vars(parser.parse_args())
    host, port, level = (args.pop(key) for key in ("host", "port", "log_level"))
    logging.basicConfig(
        level=level.upper(), format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"
    )
    uvicorn.run(create_app(ServerConfig(**args)), host=host, port=port, log_level=level, workers=1)


if __name__ == "__main__":
    main()
