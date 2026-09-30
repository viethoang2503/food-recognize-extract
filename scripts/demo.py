#!/usr/bin/env python
"""Launch the Gradio demo. A public --share link requires DEMO_USERNAME / DEMO_PASSWORD (or --no_auth)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodmm.config import add_config_args, config_from_args  # noqa: E402


def resolve_auth(share: bool, no_auth: bool) -> tuple[str, str] | None:
    user, password = os.environ.get("DEMO_USERNAME"), os.environ.get("DEMO_PASSWORD")
    if user and password and not no_auth:
        return user, password
    if share and not no_auth:
        raise SystemExit("Refusing to create a public share link without auth: set DEMO_USERNAME and "
                         "DEMO_PASSWORD (Colab Secrets), or pass --no_auth explicitly")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--no_auth", action="store_true", help="allow running without a password")
    parser.add_argument("--no_vlm", action="store_true", help="hide the VLM checkbox")
    parser.add_argument("--dry_run", action="store_true", help="check the auth settings and exit")
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    auth = resolve_auth(args.share, args.no_auth)
    if args.dry_run:
        print(f"dry run ok: share={args.share} auth={'on' if auth else 'off'}")
        return 0

    from foodmm.demo.app import build_app
    from foodmm.demo.examples import load_examples
    from foodmm.demo.predictor import DemoPredictor
    from foodmm.demo.vlm_service import LazyVLM
    from foodmm.vlm.backend import make_backend

    predictor = DemoPredictor(cfg)
    use_vlm = bool(cfg["demo"]["vlm"]) and not args.no_vlm
    vlm = LazyVLM(lambda: make_backend(cfg), cfg["vlm"]["max_image_side"]) if use_vlm else None
    app = build_app(predictor, vlm, load_examples(cfg))
    app.queue(default_concurrency_limit=1).launch(share=args.share, auth=auth,
                                                  server_port=int(cfg["demo"]["server_port"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
