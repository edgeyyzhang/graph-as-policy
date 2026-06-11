"""Frontend build artifact: dist/ is checked in and served at /."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gap.viz.server import create_app, frontend_dist_dir


def test_dist_bundle_exists() -> None:
    dist = frontend_dist_dir()
    assert dist.is_dir(), "frontend/dist must be built and checked in"
    assert (dist / "index.html").exists()
    assets = dist / "assets"
    assert any(assets.glob("*.js")), "vite build must emit a JS bundle"


def test_server_serves_index_at_root(tmp_path: Path) -> None:
    client = TestClient(create_app(root_dir=tmp_path))
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "gap viz" in res.text  # rebranded <title>

    # SPA fallback: unknown non-API path returns index.html…
    res = client.get("/some/client/route")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]

    # …but unknown API routes stay 404.
    res = client.get("/api/definitely-not-a-route")
    assert res.status_code == 404


def test_built_js_asset_served(tmp_path: Path) -> None:
    dist = frontend_dist_dir()
    js = next(iter((dist / "assets").glob("*.js")))
    client = TestClient(create_app(root_dir=tmp_path))
    res = client.get(f"/assets/{js.name}")
    assert res.status_code == 200
    assert "javascript" in res.headers["content-type"]
