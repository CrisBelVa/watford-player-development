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

- `/Users/macmontxinho/Desktop/Teams/watford/watford-player-development/scripts/run_pipeline_server.sh`

Por defecto ejecuta:

- `--mode full-db`
- `--incremental-state db`
- `--db-mode append_new`
- `--cleanup-temp`

Si quieres probarlo sin borrar temporales todavía, puedes usar:

```bash
CLEANUP_TEMP=0 RUN_MODE=incremental ./scripts/run_pipeline_server.sh
```

El script también deja un fichero de estado:

- `/Users/macmontxinho/Desktop/Teams/watford/watford-player-development/data/pipeline/server_pipeline_status.txt`

Y soporta alerta opcional por webhook si defines:

- `ALERT_WEBHOOK_URL`

## Primera ejecución en servidor

Para una carga inicial completa:

```bash
RUN_MODE=bootstrap ./scripts/run_pipeline_server.sh
```

Esto hace:

1. Descarga el histórico disponible.
2. Lo parsea.
3. Lo carga en la base.
4. Borra JSON y CSV temporales al final.

## Ejecuciones posteriores

Para actualizaciones normales:

```bash
RUN_MODE=incremental ./scripts/run_pipeline_server.sh
```

Esto hace:

1. Revisa qué partidos ya existen en la base.
2. Descarga solo los nuevos.
3. Los parsea.
4. Inserta solo lo que falta.
5. Borra los temporales.

## Modo automático

Si quieres dejar que el pipeline decida:

```bash
RUN_MODE=auto ./scripts/run_pipeline_server.sh
```

## Programación en servidor

Ejemplo de `cron` para ejecutarlo todos los martes a las 05:00:

```cron
0 5 * * 2 cd /Users/macmontxinho/Desktop/Teams/watford/watford-player-development && RUN_MODE=incremental ./scripts/run_pipeline_server.sh >> data/pipeline/cron_pipeline.log 2>&1
```

Importante:

- ese horario se interpreta en la zona horaria del servidor
- si el servidor no está en hora de Madrid, habrá que ajustar la hora

## Alerta en caso de fallo

Sí, se puede dejar una alarma si el scraping falla o no termina bien.

Con la versión actual:

1. Si falla, el script deja `status=failed` en `server_pipeline_status.txt`.
2. Si termina bien, deja `status=ok`.
3. Si defines `ALERT_WEBHOOK_URL`, envía una alerta por webhook.

Ejemplo:

```bash
export ALERT_WEBHOOK_URL="https://tu-webhook-aqui"
RUN_MODE=incremental ./scripts/run_pipeline_server.sh
```

Esto sirve bien para Slack, Make, Zapier o un endpoint propio.

## Flujo final

El flujo operativo queda así:

1. WhoScored
2. JSON temporales
3. CSV temporales
4. Base de datos cloud
5. Streamlit leyendo de la base

## Nota importante

Este modo servidor está pensado para no depender del disco local como almacenamiento permanente.

Los archivos temporales se eliminan al final porque:

- el incremental usa la base de datos como referencia
- no necesitamos conservar JSON y CSV después de una carga correcta
- reducimos uso de espacio en el servidor
