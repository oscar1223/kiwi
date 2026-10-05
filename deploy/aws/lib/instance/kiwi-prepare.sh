#!/bin/bash
# Corre como root antes de cada arranque de kiwi serve (ExecStartPre=+).
#
# 1. Vuelca el secreto de Secrets Manager al .env de kiwi, con permisos 0600.
#    Se hace en cada arranque, así que cambiar un valor del secreto es
#    `systemctl restart kiwi`, sin tocar la instancia.
# 2. La primera vez, clona el repo de trabajo con el token de GitHub.
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
    # Las claves con "_" delante son de relleno (las crea CDK); las vacías,
    # valores que aún no se han puesto.
    if key.startswith("_") or not str(value).strip():
        continue
    if "\n" in str(value):
        sys.exit(f"{key}: el valor tiene saltos de línea y no cabe en un .env")
    print(f"{key}={value}")
' >"$tmp"

for required in KIWI_TELEGRAM_TOKEN KIWI_TELEGRAM_ALLOWED_USERS; do
  if ! grep -q "^$required=" "$tmp"; then
    echo "kiwi-prepare: falta $required en el secreto $SECRET_ID" >&2
    exit 1
  fi
done

chown kiwi:kiwi "$tmp"
mv "$tmp" "$env_dir/.env"
trap - EXIT

# El clon necesita el token una vez; después lo usa el credential helper de
# ~/.gitconfig, que lo lee de GH_TOKEN en el entorno de kiwi.
if [ -n "$KIWI_REPO" ] && [ ! -d "$KIWI_WORKDIR/.git" ]; then
  gh_token=$(sed -n 's/^GH_TOKEN=//p' "$env_dir/.env")
  install -d -o kiwi -g kiwi -m 755 "$(dirname "$KIWI_WORKDIR")"
  sudo -H -u kiwi env GH_TOKEN="$gh_token" \
    git clone "https://github.com/$KIWI_REPO.git" "$KIWI_WORKDIR"
fi
install -d -o kiwi -g kiwi -m 755 "$KIWI_WORKDIR"
