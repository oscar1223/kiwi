# Two kiwi channels on the same machine:
#   kiwi      the Homebrew release, the same thing any user has
#   kiwi-dev  this code, with its own config and its own data
#
#   make dev             builds and installs kiwi-dev
#   make test            the same thing CI runs
#   make release V=x.y.z tags main and triggers the release workflow

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
	@# First time: copy profiles, keys, MCP and skills so nothing needs reconfiguring.
	@# Memory, checkpoints and sessions are not copied: they belong to the stable one.
	@if [ ! -d "$(DEV_CONFIG)" ]; then \
		mkdir -p "$(DEV_CONFIG)"; \
		for f in kiwi.json .env mcp.json; do \
			if [ -f "$(STABLE_CONFIG)/$$f" ]; then cp -p "$(STABLE_CONFIG)/$$f" "$(DEV_CONFIG)/$$f"; fi; \
		done; \
		if [ -f "$(DEV_CONFIG)/.env" ]; then chmod 600 "$(DEV_CONFIG)/.env"; fi; \
		if [ -d "$(STABLE_CONFIG)/skills" ]; then cp -Rp "$(STABLE_CONFIG)/skills" "$(DEV_CONFIG)/skills"; fi; \
		echo "Initial config copied from $(STABLE_CONFIG) to $(DEV_CONFIG)"; \
	fi
	@if [ -e "$(GOBIN)/kiwi" ]; then \
		echo "warning: $(GOBIN)/kiwi exists and shadows the Homebrew kiwi in PATH. Remove it with: rm $(GOBIN)/kiwi"; \
	fi
	@echo "kiwi-dev installed at $(DEV_BIN)"

test:
	go vet ./...
	go test ./...

release:
	@[ -n "$(V)" ] || { echo "usage: make release V=0.2.0"; exit 1; }
	@case "$(V)" in v*) echo "error: give the version without the leading v (make release V=$(patsubst v%,%,$(V)))"; exit 1;; esac
	@git fetch --quiet --tags origin
	@! git rev-parse -q --verify "refs/tags/v$(V)" >/dev/null || { echo "error: tag v$(V) already exists"; exit 1; }
	@[ "$$(git rev-parse --abbrev-ref HEAD)" = main ] || { echo "error: releases must be cut from main"; exit 1; }
	@git diff --quiet && git diff --cached --quiet || { echo "error: there are uncommitted changes"; exit 1; }
	@[ "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/main)" ] || { echo "error: main does not match origin/main (pull or push first)"; exit 1; }
	$(MAKE) test
	git tag -a "v$(V)" -m "v$(V)"
	git push origin "v$(V)"
	@echo ""
	@echo "Release v$(V) started: https://github.com/oscar1223/kiwi/actions"
	@echo "When the workflow finishes, update your stable kiwi with:"
	@echo "  brew update && brew upgrade --cask kiwi"
