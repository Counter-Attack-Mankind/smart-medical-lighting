"""Local FastAPI dashboard for the shared perception runtime."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import system_config as cfg
from src.system.perception_system import PerceptionSystem


BASE_DIR = Path(__file__).resolve().parent


class LightingModeRequest(BaseModel):
    mode: Literal["manual", "automatic"]


class ManualLightingRequest(BaseModel):
    lux: int = Field(
        ge=cfg.MIN_SIMULATED_LUX,
        le=cfg.MAX_SIMULATED_LUX,
    )


def create_app(perception_system=None, manage_lifespan=True):
    system = perception_system or PerceptionSystem()

    @asynccontextmanager
    async def lifespan(_app):
        if manage_lifespan:
            await asyncio.to_thread(system.start)
        try:
            yield
        finally:
            if manage_lifespan:
                await asyncio.to_thread(system.stop)

    application = FastAPI(
        title="Smart Medical Lighting",
        version="0.5.3",
        lifespan=lifespan,
    )
    application.state.perception_system = system
    application.mount(
        "/static",
        StaticFiles(directory=BASE_DIR / "static"),
        name="static",
    )

    @application.get("/")
    async def index():
        return FileResponse(BASE_DIR / "templates" / "index.html")

    @application.get("/api/status")
    async def status():
        return system.snapshot()

    async def mjpeg_stream(module):
        interval = 1.0 / cfg.MJPEG_TARGET_HZ
        while True:
            jpeg, _version = system.get_jpeg(module)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            await asyncio.sleep(interval)

    @application.get("/video/pose")
    async def pose_video():
        return StreamingResponse(
            mjpeg_stream("pose"),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/video/face")
    async def face_video():
        return StreamingResponse(
            mjpeg_stream("face"),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @application.websocket("/ws/monitor")
    async def monitor(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(system.snapshot())
                await asyncio.sleep(1.0 / cfg.WEBSOCKET_TARGET_HZ)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @application.post("/api/lighting/mode")
    async def set_lighting_mode(request: LightingModeRequest):
        try:
            return {"lighting": system.set_lighting_mode(request.mode)}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/api/lighting/manual")
    async def set_manual_lighting(request: ManualLightingRequest):
        try:
            return {"lighting": system.set_manual_lux(request.lux)}
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return application


app = create_app()
