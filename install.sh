#!/bin/sh
# Instalador de kiwi.
#   curl -fsSL https://raw.githubusercontent.com/oscar1223/kiwi/main/install.sh | sh
#
# Variables:
#   KIWI_VERSION      versión concreta (por defecto: la última)
#   KIWI_INSTALL_DIR  destino (por defecto: $HOME/.local/bin)
set -eu

REPO="oscar1223/kiwi"
INSTALL_DIR="${KIWI_INSTALL_DIR:-$HOME/.local/bin}"

die() { printf 'error: %s\n' "$1" >&2; exit 1; }

TMP=""
cleanup() { [ -n "$TMP" ] && rm -rf "$TMP"; :; }
trap cleanup EXIT INT TERM

# --- 1. Plataforma ---------------------------------------------------------
os=$(uname -s)
case "$os" in
  Darwin) os=darwin ;;
  Linux)  os=linux ;;
  *) die "SO no soportado: $os. Compila desde fuentes: go install github.com/$REPO/cmd/kiwi@latest" ;;
esac

arch=$(uname -m)
case "$arch" in
  x86_64|amd64)  arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) die "arquitectura no soportada: $arch. Compila desde fuentes: go install github.com/$REPO/cmd/kiwi@latest" ;;
esac

# --- 2. Descargador --------------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  fetch()  { curl -fsSL "$1" -o "$2"; }
  fetch_out() { curl -fsSL "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch()  { wget -qO "$2" "$1"; }
  fetch_out() { wget -qO- "$1"; }
else
  die "hace falta curl o wget"
fi

command -v tar >/dev/null 2>&1 || die "hace falta tar"

# --- 3. Versión ------------------------------------------------------------
version="${KIWI_VERSION:-}"
if [ -z "$version" ]; then
  version=$(fetch_out "https://api.github.com/repos/$REPO/releases/latest" \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -n 1)
  [ -n "$version" ] || die "no he podido resolver la última versión de $REPO"
fi

archive="kiwi_${os}_${arch}.tar.gz"
base="https://github.com/$REPO/releases/download/$version"

printf 'Instalando kiwi %s (%s/%s)...\n' "$version" "$os" "$arch"

# --- 4. Descarga y verificación --------------------------------------------
TMP=$(mktemp -d)
fetch "$base/$archive" "$TMP/$archive" \
  || die "no existe $archive en la release $version"
fetch "$base/checksums.txt" "$TMP/checksums.txt" \
  || die "no he podido bajar checksums.txt"

expected=$(sed -n "s/^\([0-9a-f]\{64\}\)[[:space:]][[:space:]]*$archive\$/\1/p" \
  "$TMP/checksums.txt" | head -n 1)
[ -n "$expected" ] || die "$archive no aparece en checksums.txt"

if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$TMP/$archive" | cut -d' ' -f1)
elif command -v shasum >/dev/null 2>&1; then
  actual=$(shasum -a 256 "$TMP/$archive" | cut -d' ' -f1)
else
  die "no hay sha256sum ni shasum: no puedo verificar la descarga"
fi

[ "$actual" = "$expected" ] \
  || die "checksum incorrecto (esperado $expected, obtenido $actual)"

# --- 5. Instalar -----------------------------------------------------------
tar -xzf "$TMP/$archive" -C "$TMP"
[ -f "$TMP/kiwi" ] || die "el archivo no contiene el binario kiwi"

mkdir -p "$INSTALL_DIR"
chmod +x "$TMP/kiwi"
mv "$TMP/kiwi" "$INSTALL_DIR/kiwi"

printf 'kiwi instalado en %s/kiwi\n' "$INSTALL_DIR"

# --- 6. PATH ---------------------------------------------------------------
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    printf '\n%s no está en tu PATH. Añádelo con:\n\n' "$INSTALL_DIR"
    case "${SHELL##*/}" in
      fish) printf '  fish_add_path %s\n\n' "$INSTALL_DIR" ;;
      zsh)  printf "  echo 'export PATH=\"%s:\$PATH\"' >> ~/.zshrc\n\n" "$INSTALL_DIR" ;;
      bash) printf "  echo 'export PATH=\"%s:\$PATH\"' >> ~/.bashrc\n\n" "$INSTALL_DIR" ;;
      *)    printf '  export PATH="%s:$PATH"\n\n' "$INSTALL_DIR" ;;
    esac
    ;;
esac

"$INSTALL_DIR/kiwi" --version
