#!/usr/bin/env bash
set -euo pipefail

bootstrap_runtime() {
	if [[ "${EUID}" -ne 0 ]]; then
		return
	fi

	if command -v apt-get >/dev/null 2>&1; then
		apt-get update
		apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-xetex texlive-latex-extra texlive-pstricks texlive-lang-chinese ffmpeg -y
	fi

	if ! command -v uv >/dev/null 2>&1; then
		curl -LsSf https://astral.sh/uv/install.sh | sh
	fi

	if [[ -f "$HOME/.local/bin/env" ]]; then
		# shellcheck disable=SC1090
		source "$HOME/.local/bin/env"
	fi

	if command -v uv >/dev/null 2>&1; then
		uv sync --extra med
		uv run playwright install chromium --only-shell
	fi
}

bootstrap_runtime

if command -v uv >/dev/null 2>&1; then
	exec uv run python main.py
fi

exec python main.py