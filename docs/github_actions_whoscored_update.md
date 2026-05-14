# Automatización WhoScored en GitHub Actions

## Qué hace

Se ha preparado un workflow en:

- `.github/workflows/whoscored-update.yml`

Ese workflow:

- permite ejecución manual con `workflow_dispatch`
- intenta ejecutarse cada martes
- ajusta la ejecución real a las `05:00` de `Europe/Madrid`
- usa el script existente `scripts/run_pipeline_server.sh`
- sube como artefacto el `server_pipeline_status.txt` y el `server_pipeline.log`

## Por qué hay dos horarios `cron`

GitHub Actions programa en UTC y no ajusta automáticamente el cambio de hora de Madrid.

Por eso el workflow se lanza a:

- `03:00 UTC`
- `04:00 UTC`

Después, un paso de control comprueba la hora local de Madrid y solo deja correr el pipeline cuando realmente son las `05:00` del martes.

## Secrets necesarios

Configurar en GitHub repository settings > Secrets and variables > Actions:

- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`

Opcional:

- `ALERT_WEBHOOK_URL`

## Recomendación de puesta en marcha

1. Subir el repositorio a GitHub con estos cambios.
2. Configurar los secrets.
3. Lanzar primero el workflow manualmente.
4. Revisar el artefacto y confirmar que el estado final sea `ok`.
5. Dejar activo el horario semanal.
