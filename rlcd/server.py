"""Loopback-only game service; keys and model paths never enter API responses."""
import asyncio
from contextlib import asynccontextmanager
import os
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
import httpx

from .decision import DecisionRuntime
from .protocol import ROOT

runtime = DecisionRuntime()


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(runtime.load)
    yield
    await runtime.client.aclose()


app = FastAPI(title="Cinder Station decision service", version="1.0.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])


@app.middleware("http")
async def local_requests_only(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        allowed_ports = {int(os.getenv("RLCD_WEB_PORT", "4173")), int(os.getenv("RLCD_API_PORT", "8765"))}
        try:
            valid = parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1") and parsed.port in allowed_ports
        except ValueError:
            valid = False
        if not valid:
            return JSONResponse({"detail": "Only the local game may call this service"}, status_code=403)
    return await call_next(request)


class DecisionRequest(BaseModel):
    state: str = Field(min_length=1, max_length=8000)
    mode: Literal["laya", "decoder", "jev"]
    run_id: str | None = Field(default=None, max_length=100)


@app.get("/api/status")
async def status():
    return {"ready": True, "available_modes": runtime.available_modes(), "metadata": runtime.metadata}


@app.post("/api/decision")
async def decision(request: DecisionRequest):
    if request.mode not in runtime.available_modes():
        raise HTTPException(409, "Requested model is not configured. Restart with the appropriate model option.")
    try:
        return await runtime.decide(request.state, request.mode)
    except (httpx.HTTPError, ValueError, RuntimeError, KeyError, TypeError) as error:
        # Provider bodies and exception strings may contain sensitive information.
        raise HTTPException(502, "Model request failed; the pilot stopped and no action was substituted") from error


if (ROOT / "game/dist").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "game/dist", html=True), name="game")
