"""Agent settings: which MCP servers to use and which Meta Model API model.

The provider endpoint, the model for each role (the agent uses roles.agent), reasoning effort
and per-model token pricing come from models.json next to this file (or the file named by
--model-config / $MODEL_CONFIG). Everything else comes from environment variables (the repo's
.env is loaded by the CLI):

    MODEL_API_KEY       required (Meta Model API key; the variable name is api_key_env in models.json)
    MCP_AUTH_TOKEN      shared by the agent and all three MCP servers below
    CAMERA_MCP_URL      default http://raspberrypi.local:8765/mcp   (camera_server / camera_v3)
    IMAGE_MCP_URL       default http://127.0.0.1:10001/mcp          (image_server / mcp_image_server)
    TRACKING_MCP_URL    default http://127.0.0.1:10002/mcp          (tracking_mcp, inside existing_work_before_hackathon/detection_ingest_server.py)

Set a URL to an empty string to leave that server out (run_demo.py runs with images only).

Or pass --servers servers.json with a list of {"name", "url", "token"} objects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_MODEL_CONFIG = os.path.join(os.path.dirname(__file__), "models.json")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SAVE_DIR = os.path.join(REPO_ROOT, "output", "agent_runs")


@dataclass
class ServerConfig:
    name: str  # used as the tool-name prefix, e.g. "camera" -> camera__capture_frame
    url: str
    token: str | None = None
    timeout_s: float = 60.0


@dataclass
class AgentConfig:
    model: str = ""
    base_url: str = "https://api.meta.ai/v1"  # Meta Model API (OpenAI-compatible Responses API)
    api_key_env: str = "MODEL_API_KEY"
    servers: list[ServerConfig] = field(default_factory=list)
    max_steps: int = 12  # model turns before giving up
    image_detail: str = "auto"  # "low" | "high" | "auto" for images sent to the model
    reasoning_effort: str | None = None  # e.g. "low", "medium", "high" for reasoning models
    save_dir: str | None = DEFAULT_SAVE_DIR
    # model id -> {"input", "cached_input", "output"} in USD per 1M tokens (for the cost estimate)
    pricing: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, servers_file: str | None = None, model_file: str | None = None) -> "AgentConfig":
        with open(model_file or os.getenv("MODEL_CONFIG") or DEFAULT_MODEL_CONFIG) as f:
            models = json.load(f)
        if servers_file:
            with open(servers_file) as f:
                servers = [ServerConfig(**s) for s in json.load(f)]
        else:
            token = os.getenv("MCP_AUTH_TOKEN") or None  # one token for every MCP server
            urls = {
                "camera": os.getenv("CAMERA_MCP_URL", "http://raspberrypi.local:8765/mcp"),
                "images": os.getenv("IMAGE_MCP_URL", "http://127.0.0.1:10001/mcp"),
                "tracking": os.getenv("TRACKING_MCP_URL", "http://127.0.0.1:10002/mcp"),
            }
            servers = [ServerConfig(name, url.strip(), token) for name, url in urls.items() if url.strip()]
        return cls(
            model=models["roles"]["agent"]["model"],
            base_url=models["base_url"],
            api_key_env=models["api_key_env"],
            servers=servers,
            reasoning_effort=models.get("reasoning_effort") or None,
            pricing=models.get("pricing_usd_per_1m_tokens", {}),
        )

    def estimate_cost_usd(self, usage: dict, model: str | None = None) -> float | None:
        """Short-context Standard-tier estimate; None if the model has no pricing entry."""
        p = self.pricing.get(model or self.model)
        if p is None:
            return None
        cached = usage.get("cached_tokens", 0)
        uncached = usage.get("input_tokens", 0) - cached
        return (uncached * p["input"] + cached * p.get("cached_input", p["input"]) + usage.get("output_tokens", 0) * p["output"]) / 1_000_000
