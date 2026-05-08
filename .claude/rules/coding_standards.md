# Coding Standards

## Backend
- FastAPI
- Modular routers
- No business logic in routes

## Structure
api/ → endpoints
core/ → config + infra
services/ → logic
models/ → schemas

## Rules
- Use type hints everywhere
- Avoid global state