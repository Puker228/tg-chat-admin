format:
	uv run ruff check --select I,F401 --fix
	uv run ruff format

up-dev:
	uv run python manage.py runserver

test-local:
	uv run python manage.py test

up:
	docker compose build
	docker compose down
	docker compose up -d

logs:
	docker compose logs -f bot

migration-create:
	uv run python manage.py makemigrations

migration-run:
	uv run python manage.py migrate