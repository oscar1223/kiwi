"""terminal-bench installed-agent adapter for kiwi (github.com/oscar1223/kiwi).

kiwi has no package-manager release, so this adapter ships a prebuilt static
Linux binary (see bench/build-dist.sh) and copies it into the task container
via terminal-bench's copy_to_container, instead of running `go install` at
trial time. That copy goes through the Docker daemon API (put_archive), not
the container's own network stack, so it also works on tasks that start with
broken networking inside the container (e.g. cron-broken-network) — unlike a
`go install`-based install script, which needs the container's network up
just to fetch the agent itself.

Model selection in kiwi is done via named profiles in
~/.config/kiwi/kiwi.json (sonnet/opus/gpt/local), not an arbitrary
"provider/model" string, so this adapter maps terminal-bench's --model into
one of kiwi's built-in default profiles rather than passing it through
verbatim.

Usage:
    bench/build-dist.sh   # once, or whenever kiwi's source changes
    tb run --agent-import-path tb_kiwi_agent.kiwi_agent:KiwiAgent \
        -k profile=gpt ...
"""

import os
import platform
import shlex
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult
from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand
from terminal_bench.terminal.tmux_session import TmuxSession

# Docker Desktop runs containers natively on the host architecture, so the
# binary we copy in just needs to match the host, not anything task-specific.
_GOARCH = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}
_DIST_DIR = Path(__file__).resolve().parent.parent / "dist"

# kiwi's built-in default profiles (see internal/config.Default in the kiwi
# repo) and which env var each one needs.
_PROFILE_ENV = {
    "sonnet": "ANTHROPIC_API_KEY",
    "opus": "ANTHROPIC_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class KiwiAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return "kiwi"

    def __init__(self, model_name: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name

        arch = _GOARCH.get(platform.machine())
        if arch is None:
            raise ValueError(
                f"unsupported host architecture {platform.machine()!r}; "
                "KiwiAgent ships prebuilt linux/amd64 and linux/arm64 binaries only"
            )
        self._binary_path = _DIST_DIR / f"kiwi-linux-{arch}"
        if not self._binary_path.exists():
            raise FileNotFoundError(
                f"{self._binary_path} not found — build it first with "
                "bench/build-dist.sh"
            )

        profile = kwargs.get("profile")
        if profile is None:
            if os.environ.get("ANTHROPIC_API_KEY"):
                profile = "sonnet"
            elif os.environ.get("OPENAI_API_KEY"):
                profile = "gpt"
            else:
                raise ValueError(
                    "KiwiAgent needs ANTHROPIC_API_KEY or OPENAI_API_KEY in the "
                    "environment (or pass -k profile=<sonnet|opus|gpt> explicitly)"
                )
        if profile not in _PROFILE_ENV:
            raise ValueError(
                f"Unknown kiwi profile {profile!r}; expected one of "
                f"{sorted(_PROFILE_ENV)}"
            )
        self._profile = profile

    @property
    def _env(self) -> dict[str, str]:
        key = _PROFILE_ENV[self._profile]
        if key not in os.environ:
            raise ValueError(
                f"kiwi profile {self._profile!r} needs {key} in the environment"
            )
        return {key: os.environ[key]}

    @property
    def _install_agent_script_path(self) -> Path:
        return self._get_templated_script_path("kiwi-setup.sh.j2")

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        # Copy the prebuilt binary in before the install script runs. This
        # goes through the Docker daemon API, not the container's network
        # stack, so it works even on tasks that start with broken networking.
        session.copy_to_container(
            self._binary_path,
            container_dir="/installed-agent",
            container_filename="kiwi",
        )
        return super().perform_task(instruction, session, logging_dir)

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped_instruction = shlex.quote(instruction)
        return [
            TerminalCommand(
                command=(
                    f"kiwi ask -m {self._profile} --mode work --yolo -q "
                    f"{escaped_instruction}"
                ),
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]
