#!/bin/sh
# kiwi installer.
#   curl -fsSL https://raw.githubusercontent.com/oscar1223/kiwi/main/install.sh | sh
#
# Variables:
#   KIWI_VERSION      specific version (default: the latest)
#   KIWI_INSTALL_DIR  destination (default: $HOME/.local/bin)
set -eu

REPO="oscar1223/kiwi"
INSTALL_DIR="${KIWI_INSTALL_DIR:-$HOME/.local/bin}"

die() { printf 'error: %s\n' "$1" >&2; exit 1; }

TMP=""
cleanup() { [ -n "$TMP" ] && rm -rf "$TMP"; :; }
trap cleanup EXIT INT TERM

# --- 1. Platform -----------------------------------------------------------
os=$(uname -s)
case "$os" in
  Darwin) os=darwin ;;
  Linux)  os=linux ;;
  *) die "unsupported OS: $os. Build from source: go install github.com/$REPO/cmd/kiwi@latest" ;;
esac

arch=$(uname -m)
case "$arch" in
  x86_64|amd64)  arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) die "unsupported architecture: $arch. Build from source: go install github.com/$REPO/cmd/kiwi@latest" ;;
esac

# --- 2. Downloader ---------------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  fetch()  { curl -fsSL "$1" -o "$2"; }
  fetch_out() { curl -fsSL "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch()  { wget -qO "$2" "$1"; }
  fetch_out() { wget -qO- "$1"; }
else
  die "curl or wget is required"
fi

command -v tar >/dev/null 2>&1 || die "tar is required"

# --- 3. Version ------------------------------------------------------------
version="${KIWI_VERSION:-}"
if [ -z "$version" ]; then
  version=$(fetch_out "https://api.github.com/repos/$REPO/releases/latest" \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -n 1)
  [ -n "$version" ] || die "could not resolve the latest version of $REPO"
fi

archive="kiwi_${os}_${arch}.tar.gz"
base="https://github.com/$REPO/releases/download/$version"

printf 'Installing kiwi %s (%s/%s)...\n' "$version" "$os" "$arch"

# --- 4. Download and verification -----------------------------------------
TMP=$(mktemp -d)
fetch "$base/$archive" "$TMP/$archive" \
  || die "$archive does not exist in release $version"
fetch "$base/checksums.txt" "$TMP/checksums.txt" \
  || die "no he podido bajar checksums.txt"

expected=$(sed -n "s/^\([0-9a-f]\{64\}\)[[:space:]][[:space:]]*$archive\$/\1/p" \
  "$TMP/checksums.txt" | head -n 1)
[ -n "$expected" ] || die "$archive is not listed in checksums.txt"

if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$TMP/$archive" | cut -d' ' -f1)
elif command -v shasum >/dev/null 2>&1; then
  actual=$(shasum -a 256 "$TMP/$archive" | cut -d' ' -f1)
else
  die "neither sha256sum nor shasum found: cannot verify the download"
fi

[ "$actual" = "$expected" ] \
  || die "checksum incorrecto (esperado $expected, obtenido $actual)"

# --- 5. Instalar -----------------------------------------------------------
tar -xzf "$TMP/$archive" -C "$TMP"
[ -f "$TMP/kiwi" ] || die "the archive does not contain the kiwi binary"

mkdir -p "$INSTALL_DIR"
chmod +x "$TMP/kiwi"
mv "$TMP/kiwi" "$INSTALL_DIR/kiwi"

printf 'kiwi installed at %s/kiwi\n' "$INSTALL_DIR"

# --- 6. PATH ---------------------------------------------------------------
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    printf '\n%s is not in your PATH. Add it with:\n\n' "$INSTALL_DIR"
    case "${SHELL##*/}" in
      fish) printf '  fish_add_path %s\n\n' "$INSTALL_DIR" ;;
      zsh)  printf "  echo 'export PATH=\"%s:\$PATH\"' >> ~/.zshrc\n\n" "$INSTALL_DIR" ;;
      bash) printf "  echo 'export PATH=\"%s:\$PATH\"' >> ~/.bashrc\n\n" "$INSTALL_DIR" ;;
      *)    printf '  export PATH="%s:$PATH"\n\n' "$INSTALL_DIR" ;;
    esac
    ;;
esac

"$INSTALL_DIR/kiwi" --version
