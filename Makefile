# Makefile for managing the StreamViz STAC API stack

ENV_FILE=.env
COMPOSE=docker compose --env-file $(ENV_FILE)
PGDATA_DIR=$(shell grep PG_DATA $(ENV_FILE) | cut -d '=' -f2)

# Default target
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  up           Start all containers (in foreground)"
	@echo "  up-detached  Start all containers (in background)"
	@echo "  down         Stop and remove all containers"
	@echo "  logs         Tail logs from all services"
	@echo "  api-logs     Tail logs from stac-api only"
	@echo "  build        Build images (with cache)"
	@echo "  rebuild      Build images (no cache)"
	@echo "  restart      Stop and restart services"
	@echo "  reset-db     Force delete DB volume & start fresh"
	@echo "  status       Show running containers"
	@echo "  curl         Test STAC API root endpoint"

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
	curl -s http://localhost:$(shell grep STAC_API_PORT $(ENV_FILE) | cut -d '=' -f2) | jq .

.PHONY: reset-db
reset-db: down
	@if [ -d "$(PGDATA_DIR)" ]; then \
		echo "Removing volume directory $(PGDATA_DIR)..."; \
		sudo rm -rf $(PGDATA_DIR); \
	else \
		echo "No volume directory found at $(PGDATA_DIR)"; \
	fi
	$(MAKE) up-detached

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
	@echo "Open: http://localhost:$(shell grep DASH_PORT $(ENV_FILE) | cut -d '=' -f2)"

.PHONY: dash-logs
dash-logs:
	docker logs -f plot-dash

.PHONY: services
services:
	docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"




# make up           # Run stack interactively
# make up-detached  # Run stack in background
# make down         # Stop all containers
# make logs         # Follow logs from all services
# make api-logs     # Just stac-api logs
# make rebuild      # Rebuild everything from scratch
# make reset-db     # Delete volumes and restart clean
# make curl         # Hit http://localhost:8081 and print JSON


# make build        # build (adds the dash image)
# make up-detached  # bring the stack up (including your existing services)
# make dash-up      # or just start Dash if the rest is already running
# make dash-open    # open the UI
# -> http://localhost:8050

