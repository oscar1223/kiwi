# Dos canales de kiwi en la misma máquina:
#   kiwi      la release de Homebrew, lo mismo que tiene cualquier usuario
#   kiwi-dev  este código, con su propia config y sus propios datos
#
#   make dev             compila e instala kiwi-dev
#   make test            lo mismo que ejecuta la CI
#   make release V=x.y.z etiqueta main y lanza el workflow de release

GOBIN := $(shell go env GOBIN)
ifeq ($(GOBIN),)
GOBIN := $(shell go env GOPATH)/bin
endif

PKG           := github.com/oscar1223/kiwi
DEV_BIN       := $(GOBIN)/kiwi-dev
CONFIG_HOME   := $(or $(XDG_CONFIG_HOME),$(HOME)/.config)
STABLE_CONFIG := $(CONFIG_HOME)/kiwi
DEV_CONFIG    := $(CONFIG_HOME)/kiwi-dev

.PHONY: dev test release

dev:
	go build -ldflags "-X main.version=dev -X $(PKG)/internal/config.appName=kiwi-dev" -o "$(DEV_BIN)" ./cmd/kiwi
	@# Primera vez: copia perfiles, claves, MCP y skills para no reconfigurar.
	@# La memoria, los checkpoints y las sesiones no se copian: son del estable.
	@if [ ! -d "$(DEV_CONFIG)" ]; then \
		mkdir -p "$(DEV_CONFIG)"; \
		for f in kiwi.json .env mcp.json; do \
			if [ -f "$(STABLE_CONFIG)/$$f" ]; then cp -p "$(STABLE_CONFIG)/$$f" "$(DEV_CONFIG)/$$f"; fi; \
		done; \
		if [ -f "$(DEV_CONFIG)/.env" ]; then chmod 600 "$(DEV_CONFIG)/.env"; fi; \
		if [ -d "$(STABLE_CONFIG)/skills" ]; then cp -Rp "$(STABLE_CONFIG)/skills" "$(DEV_CONFIG)/skills"; fi; \
		echo "Config inicial copiada de $(STABLE_CONFIG) a $(DEV_CONFIG)"; \
	fi
	@if [ -e "$(GOBIN)/kiwi" ]; then \
		echo "aviso: $(GOBIN)/kiwi existe y tapa al kiwi de Homebrew en el PATH. Bórralo con: rm $(GOBIN)/kiwi"; \
	fi
	@echo "kiwi-dev instalado en $(DEV_BIN)"

test:
	go vet ./...
	go test ./...

release:
	@[ -n "$(V)" ] || { echo "uso: make release V=0.2.0"; exit 1; }
	@case "$(V)" in v*) echo "error: pon la versión sin la v inicial (make release V=$(patsubst v%,%,$(V)))"; exit 1;; esac
	@git fetch --quiet --tags origin
	@! git rev-parse -q --verify "refs/tags/v$(V)" >/dev/null || { echo "error: el tag v$(V) ya existe"; exit 1; }
	@[ "$$(git rev-parse --abbrev-ref HEAD)" = main ] || { echo "error: hay que publicar desde main"; exit 1; }
	@git diff --quiet && git diff --cached --quiet || { echo "error: hay cambios sin commitear"; exit 1; }
	@[ "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/main)" ] || { echo "error: main no coincide con origin/main (haz pull o push primero)"; exit 1; }
	$(MAKE) test
	git tag -a "v$(V)" -m "v$(V)"
	git push origin "v$(V)"
	@echo ""
	@echo "Release v$(V) lanzada: https://github.com/oscar1223/kiwi/actions"
	@echo "Cuando termine el workflow, actualiza tu kiwi estable con:"
	@echo "  brew update && brew upgrade --cask kiwi"
