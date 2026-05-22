#!/usr/bin/env bash
set -euo pipefail

declare -a child_pids=()

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

run_service() {
	if command -v uv >/dev/null 2>&1; then
		uv run "$@"
	else
		"$@"
	fi
}

cleanup() {
	for pid in "${child_pids[@]:-}"; do
		if [[ -n "$pid" ]]; then
			kill "$pid" 2>/dev/null || true
		fi
	done
}

trap cleanup EXIT INT TERM

bootstrap_runtime

if [[ "${WEBADMIN_ENABLED:-1}" != "0" ]]; then
	run_service miobot-webadmin &
	child_pids+=("$!")
fi

if [[ "${BOT_ENABLED:-1}" != "0" ]]; then
	run_service python main.py &
	child_pids+=("$!")
fi

if [[ "${#child_pids[@]}" -eq 0 ]]; then
	echo "No services enabled. Set BOT_ENABLED=1 or WEBADMIN_ENABLED=1." >&2
	exit 1
fi

set +e
wait -n "${child_pids[@]}"
exit_code=$?
set -e

cleanup
wait || true
exit "$exit_code"