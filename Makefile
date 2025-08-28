# Makefile for managing the StreamViz STAC API stack

ENV_FILE := .env
COMPOSE  := docker compose --env-file $(ENV_FILE)

# Vars from .env
PGDATA_DIR     := $(shell grep ^PG_DATA $(ENV_FILE) | cut -d '=' -f2)
STAC_PORT      := $(shell grep ^STAC_API_PORT $(ENV_FILE) | cut -d '=' -f2)
DASH_PORT      := $(shell grep ^DASH_PORT $(ENV_FILE) | cut -d '=' -f2)
BROWSER_PORT   := $(shell grep ^BROWSER_PORT $(ENV_FILE) | cut -d '=' -f2 2>/dev/null)

PGUSER         := $(shell grep ^POSTGRES_USER $(ENV_FILE) | cut -d '=' -f2)
PGPASSWORD     := $(shell grep ^POSTGRES_PASSWORD $(ENV_FILE) | cut -d '=' -f2)
PGDATABASE     := $(shell grep ^POSTGRES_DB $(ENV_FILE) | cut -d '=' -f2)
PGPORT         := $(shell grep ^POSTGRES_PORT $(ENV_FILE) | cut -d '=' -f2)

# DSN used by pypgstac inside the stac-api container (db is service DNS)
DSN            := postgresql://$(PGUSER):$(PGPASSWORD)@db:$(PGPORT)/$(PGDATABASE)

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Core:"
	@echo "  up              Start all containers (foreground)"
	@echo "  up-detached     Start all containers (background)"
	@echo "  down            Stop and remove all containers"
	@echo "  restart         Stop and restart services"
	@echo "  build           Build images (with cache)"
	@echo "  rebuild         Build images (no cache)"
	@echo "  status          Show running containers"
	@echo "  logs            Tail logs from all services"
	@echo "  api-logs        Tail logs from stac-api only"
	@echo "  curl            Test STAC API root endpoint"
	@echo "  reset-db        Delete DB volume folder & restart fresh"
	@echo ""
	@echo "Dash:"
	@echo "  dash-up         Start Dash only"
	@echo "  dash-down       Stop/remove Dash only"
	@echo "  dash-restart    Restart Dash"
	@echo "  dash-logs       Tail Dash logs"
	@echo "  dash-open       Print Dash URL"
	@echo ""
	@echo "STAC Browser:"
	@echo "  browser-up      Start STAC Browser only"
	@echo "  browser-down    Stop/remove STAC Browser only"
	@echo "  browser-logs    Tail STAC Browser logs"
	@echo "  browser-open    Print STAC Browser URL with API preloaded"
	@echo ""
	@echo "DB Ops:"
	@echo "  db-wait         Wait until Postgres is ready"
	@echo "  api-wait        Wait until STAC API responds"
	@echo "  db-shell        Open psql shell in DB container"
	@echo "  db-migrate      Run pypgstac migrations to head"
	@echo "  load-stac FILE=path/to/catalog.json"
	@echo "                  Load a catalog/collection/item via pypgstac"

# ── Core ──────────────────────────────────────────────────────────────────────

.PHONY: up
up:
	$(COMPOSE) up

.PHONY: up-detached
up-detached:
	$(COMPOSE) up -d

.PHONY: down
down:
	$(COMPOSE) down --remove-orphans

.PHONY: logs
logs:
	$(COMPOSE) logs -f

.PHONY: api-logs
api-logs:
	$(COMPOSE) logs -f stac-api

.PHONY: build
build:
	$(COMPOSE) build

.PHONY: rebuild
rebuild:
	$(COMPOSE) build --no-cache

.PHONY: restart
restart: down up-detached

.PHONY: status
status:
	$(COMPOSE) ps

.PHONY: curl
curl:
	@curl -s http://localhost:$(STAC_PORT) | jq .

.PHONY: reset-db
reset-db: down
	@if [ -d "$(PGDATA_DIR)" ]; then \
		echo "Removing volume directory $(PGDATA_DIR)..."; \
		sudo rm -rf $(PGDATA_DIR); \
	else \
		echo "No volume directory found at $(PGDATA_DIR)"; \
	fi
	$(MAKE) up-detached

# ── Dash ──────────────────────────────────────────────────────────────────────

.PHONY: dash-up dash-down dash-logs dash-restart dash-open
dash-up:
	$(COMPOSE) up -d dash

dash-down:
	$(COMPOSE) rm -sf dash

dash-logs:
	$(COMPOSE) logs -f dash

dash-restart:
	$(MAKE) dash-down
	$(MAKE) dash-up

dash-open:
	@echo "Open: http://localhost:$(DASH_PORT)"

# ── STAC Browser ─────────────────────────────────────────────────────────────

.PHONY: browser-up browser-down browser-logs browser-open
browser-up:
	$(COMPOSE) up -d stac-browser

browser-down:
	$(COMPOSE) rm -sf stac-browser

browser-logs:
	$(COMPOSE) logs -f stac-browser

.PHONY: browser-open
browser-open:
	@echo "Open: http://localhost:$(BROWSER_PORT)/#/http://localhost:$(STAC_API_PORT)/"

# ── DB / API Utilities ────────────────────────────────────────────────────────

.PHONY: db-wait api-wait db-shell db-migrate load-stac
db-wait:
	@echo "Waiting for Postgres @ db:$(PGPORT) ..."
	@$(COMPOSE) exec -T db bash -lc 'until pg_isready -U $(PGUSER) -d $(PGDATABASE) -h 127.0.0.1 -p $(PGPORT); do sleep 1; done; echo "Postgres ready."'

api-wait:
	@echo "Waiting for STAC API @ http://localhost:$(STAC_PORT) ..."
	@bash -lc 'until curl -fsS "http://localhost:$(STAC_PORT)/"; do sleep 1; done; echo "STAC API ready."'

db-shell:
	$(COMPOSE) exec db psql -U $(PGUSER) -d $(PGDATABASE)

db-migrate: db-wait
	@echo "Running pypgstac migrations to head ..."
	# Try new cmd first, then fallback for other versions
	@$(COMPOSE) exec -T stac-api bash -lc 'pypgstac migrate-to-head --dsn "$(DSN)" || pypgstac migrate-to-latest --dsn "$(DSN)"'
	@echo "Migration complete."

# Usage: make load-stac FILE=path/to/catalog_or_items.json
load-stac: db-wait
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Provide FILE=... (catalog/collection/item JSON)"; \
		exit 1; \
	fi
	@echo "Loading $(FILE) into $(DSN) ..."
	@$(COMPOSE) exec -T stac-api bash -lc 'pypgstac load --dsn "$(DSN)" -f "/app/$(FILE)"'
	@echo "Load complete."

# ── Misc ──────────────────────────────────────────────────────────────────────

.PHONY: services
services:
	docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

.PHONY: clean-containers
clean-containers:
	@echo "Removing ALL running/stopped containers not managed by docker-compose..."
	@docker rm -f $$(docker ps -aq) 2>/dev/null || true
	@echo "✅ All stray containers removed."


# Quick tips:
# make up            # run stack in foreground
# make up-detached   # run stack in background
# make reset-db      # nuke volumes folder & restart clean
# make db-migrate    # run pypgstac migrations (required once)
# make browser-up    # start STAC Browser
# make browser-open  # echo Browser URL preloaded with the API
# make load-stac FILE=seed/catalog.json
