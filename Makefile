format:
	uv run ruff check --select I,F401 --fix
	uv run ruff format

up-dev:
	uv run python manage.py runserver

test-local:
	uv run python manage.py test

up:
	if docker network ls --format "{{.Name}}" | grep billagorilla_network; then \
		echo "Network 'billagorilla_network' already exists"; \
	else \
		echo "Creating network 'billagorilla_network'..."; \
		docker network create billagorilla_network; \
	fi

	docker compose build
	docker compose down
	docker compose up -d

migration-create:
	uv run python manage.py makemigrations

migration-run:
	uv run python manage.py migrate