# Pipeline Modo Servidor

Este documento resume el flujo recomendado para ejecutar el scraping en un servidor sin depender de archivos locales persistentes.

## Objetivo

El servidor debe:

1. Descargar datos desde WhoScored.
2. Convertir los JSON a CSV temporalmente.
3. Cargar solo los datos nuevos en la base de datos.
4. Eliminar los temporales al terminar.

La base de datos queda como fuente principal de verdad.

## Variables de entorno necesarias

El proceso usa las mismas variables que la app:

- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`

Si existe un archivo `.env` en la raíz del proyecto, el script lo carga automáticamente.

## Script preparado

Se ha añadido este script:

- `scripts/run_pipeline_server.sh`

Por defecto ejecuta:

- `--mode full-db`
- `--incremental-state db`
- `--db-mode append_new`
- `--cleanup-temp`

Si quieres probarlo sin borrar temporales todavía, puedes usar:

```bash
CLEANUP_TEMP=0 RUN_MODE=incremental ./scripts/run_pipeline_server.sh
```
