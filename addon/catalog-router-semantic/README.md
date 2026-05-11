# Catalog Router Semantic Service

This Home Assistant add-on provides semantic intent ranking for the
`catalog_conversation_router` integration.

It exposes a small HTTP API on port `8099`:

- `GET /health`
- `POST /rank/phrase`
- `POST /rank/entity`

## Integration setup

After installing the add-on, set the integration option
`semantic_service_url` to the URL Home Assistant can use to reach the
service.

Examples:

- `http://homeassistant.local:8099`
- `http://<your-ha-host-ip>:8099`

The exact reachable URL depends on your Home Assistant deployment and network
setup.
