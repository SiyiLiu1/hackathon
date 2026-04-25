"""FastAPI entrypoint for governance-aware safe-view APIs."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes.clients import router as clients_router


app = FastAPI(
    title="BuildersVault Safe View API",
    version="0.1.0",
    description="Policy-gated safe-view endpoints for referral coordination demos.",
)

# TODO: Add authentication/authorization integration before production rollout.
app.include_router(clients_router)

