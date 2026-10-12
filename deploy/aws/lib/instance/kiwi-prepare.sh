#!/bin/bash
# Runs as root before every start of kiwi serve (ExecStartPre=+).
#
# 1. Dumps the Secrets Manager secret into kiwi's .env, with mode 0600.
#    It happens on every start, so changing a value in the secret is just
#    `systemctl restart kiwi`, without touching the instance.
# 2. The first time, clones the working repo with the GitHub token.
set -euo pipefail

# shellcheck source=/dev/null
source /etc/kiwi/serve.conf # SECRET_ID, AWS_REGION, KIWI_REPO, KIWI_WORKDIR

home=/home/kiwi
env_dir="$home/.config/kiwi"
install -d -o kiwi -g kiwi -m 700 "$home/.config" "$env_dir"

umask 077
tmp=$(mktemp "$env_dir/.env.XXXXXX")
trap 'rm -f "$tmp"' EXIT

aws secretsmanager get-secret-value \
  --region "$AWS_REGION" --secret-id "$SECRET_ID" \
  --query SecretString --output text |
  python3 -c '
import json, sys
for key, value in json.load(sys.stdin).items():
    # Keys starting with "_" are placeholders (CDK creates them); empty ones
    # are values that have not been set yet.
    if key.startswith("_") or not str(value).strip():
        continue
    if "\n" in str(value):
        sys.exit(f"{key}: the value has line breaks and does not fit in a .env")
    print(f"{key}={value}")
' >"$tmp"

for required in KIWI_TELEGRAM_TOKEN KIWI_TELEGRAM_ALLOWED_USERS; do
  if ! grep -q "^$required=" "$tmp"; then
    echo "kiwi-prepare: $required is missing from secret $SECRET_ID" >&2
    exit 1
  fi
done

chown kiwi:kiwi "$tmp"
mv "$tmp" "$env_dir/.env"
trap - EXIT

# The clone needs the token once; after that the credential helper in
# ~/.gitconfig uses it, reading it from GH_TOKEN in kiwi's environment.
if [ -n "$KIWI_REPO" ] && [ ! -d "$KIWI_WORKDIR/.git" ]; then
  gh_token=$(sed -n 's/^GH_TOKEN=//p' "$env_dir/.env")
  install -d -o kiwi -g kiwi -m 755 "$(dirname "$KIWI_WORKDIR")"
  sudo -H -u kiwi env GH_TOKEN="$gh_token" \
    git clone "https://github.com/$KIWI_REPO.git" "$KIWI_WORKDIR"
fi
install -d -o kiwi -g kiwi -m 755 "$KIWI_WORKDIR"
