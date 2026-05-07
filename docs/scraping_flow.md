# Watford Scouting App

## Flujo de Scraping

Este documento explica cómo funciona actualmente el flujo de scraping dentro del repositorio, qué hace cada bloque y cómo encaja la nueva capa de pipeline.

## 1. Objetivo principal

La pila de scraping está pensada para:

- descubrir los fixtures de las competiciones seleccionadas
- descargar los JSON brutos de los partidos
- transformar esos JSON en tablas estructuradas
- dejar esos outputs listos para una futura carga en el modelo de datos de la app

Hoy el proceso sigue siendo en parte exploratorio y orientado a notebooks, pero el repositorio ya contiene suficiente código reutilizable como para ejecutarlo como pipeline scriptado.

## 2. Zonas principales del repositorio implicadas

Espacio principal de scraping:

- `prueba lucas/`

Notebooks importantes:

- `prueba lucas/Scrap_WS.ipynb`
- `prueba lucas/Scrapping_Championship_Data.ipynb`
- `prueba lucas/matches_championship_download.ipynb`
- `prueba lucas/20250928_integrador.ipynb`

Módulos legacy de scraping:

- `prueba lucas/whoscored_scraping_fun.py`
- `prueba lucas/sw_scraping_fun.py`

Módulos legacy de parseo JSON:

- `prueba lucas/whoscored_json2csv_fun.py`
- `prueba lucas/sw_json2csv_fun.py`

Configuración y datos auxiliares:

- `prueba lucas/config/metadata.xlsx`
- `prueba lucas/config/prov_params.json`
- `prueba lucas/fixtures/`

Ficheros brutos descargados:

- `prueba lucas/whoscored/`

Outputs parseados:

- `prueba lucas/whoscored/output/`

## 3. Flujo funcional actual

El flujo actual tiene cuatro grandes etapas.

### Etapa A. Descubrimiento de fixtures

Objetivo:

- obtener el listado de partidos que se van a monitorizar o descargar

En el repo hay dos familias de fuentes.

#### A1. Flujo WhoScored

Archivo principal:

- `prueba lucas/whoscored_scraping_fun.py`

Función clave:

- `scrape_fixtures(fixtures_url, mes_ini=200001)`

Cómo funciona:

- abre la página de fixtures con Selenium
- navega mes a mes por el calendario
- extrae metadatos de partidos desde el HTML renderizado
- construye un DataFrame con id de partido, fecha, equipos, marcador, cuotas y metadatos de competición

Output típico:

- un DataFrame de fixtures que luego se usa para decidir qué hay que descargar

#### A2. Flujo Scoresway

Archivo principal:

- `prueba lucas/sw_scraping_fun.py`

Funciones clave:

- `obtener_sdapi_outlet_key(url_competicion)`
- `obtener_fixture_json(sdapi_outlet_key, torneo_id, callback_id, url_competicion)`
- `generar_dataframe_desde_competicion(url_competicion, fixture_json, metadata)`
- `scrape_fixtures(metadata, url_competicion)`

Cómo funciona:

- lee la página de competición
- extrae la `sdapi_outlet_key`
- consulta los feeds de Perform en formato JSONP
- normaliza la respuesta en un DataFrame de fixtures

Output típico:

- un DataFrame con el mismo propósito funcional que el de WhoScored

### Etapa B. Descarga de partidos en bruto

Objetivo:

- descargar un fichero JSON bruto por cada partido jugado

#### B1. Descarga bruta desde WhoScored

Archivo principal:

- `prueba lucas/whoscored_scraping_fun.py`

Función clave:

- `get_json_games(df, ruta, ini=200001)`

Cómo funciona:

- filtra fixtures por fecha
- salta partidos ya existentes localmente y aparentemente completos
- abre cada partido en Selenium
- busca el script embebido que contiene `matchCentreData`
- reconstruye el payload JSON
- guarda un `.json` por partido

Output:

- ficheros JSON brutos en una carpeta como `prueba lucas/whoscored/`

#### B2. Descarga bruta desde Scoresway

Archivo principal:

- `prueba lucas/sw_scraping_fun.py`

Función clave:

- `get_json_games(df_partidos, ruta_dest, url_competicion)`

Cómo funciona:

- recorre las filas del DataFrame de fixtures
- ignora partidos aún no jugados
- llama al endpoint de eventos de Perform
- elimina la envoltura JSONP
- guarda el payload JSON en disco

Output:

- ficheros JSON brutos en el directorio de destino

### Etapa C. Conversión de JSON a tablas

Objetivo:

- transformar los JSON brutos de partido en tablas de fútbol estructuradas

#### C1. Parseo WhoScored

Archivo principal:

- `prueba lucas/whoscored_json2csv_fun.py`

Responsabilidades principales:

- leer ficheros JSON brutos
- extraer información a nivel partido
- extraer equipos y estadísticas agregadas de equipo
- extraer jugadores y estadísticas agregadas de jugador
- extraer eventos a nivel acción
- crear tablas estándar como:
  - `eventData`
  - `playerData`
  - `playerStats`
  - `teamData`
  - `teamStats`
  - `matchData`

Punto de entrada útil:

- `procesar_ficheros_lista(ruta, subr)`

Esta función recorre el directorio de JSON brutos y exporta CSVs parseados en la subcarpeta `output`.

#### C2. Parseo Scoresway

Archivo principal:

- `prueba lucas/sw_json2csv_fun.py`

Responsabilidades principales:

- leer ficheros JSON brutos de eventos
- extraer `matchData`
- extraer `teamData`
- extraer `playerData`
- extraer `eventData`

Punto de entrada útil:

- `procesar_ficheros_lista(ruta, subr)`

### Etapa D. Consumo por analítica o por la app

Objetivo:

- hacer que los datos parseados sean utilizables para analítica y, finalmente, para la app Streamlit

Hoy esta etapa sigue siendo parcialmente manual y exploratoria:

- los notebooks inspeccionan outputs
- se generan y reutilizan CSVs
- algunos módulos de la app leen tablas de base de datos compatibles con este esquema futbolístico

La dirección recomendada a futuro es:

- descargar y parsear automáticamente
- consolidar outputs
- cargar tablas finales en MySQL
- hacer que Streamlit lea siempre del estado más reciente de la base

## 4. Por qué se han usado notebooks

El enfoque basado en notebooks tiene sentido para el nivel actual de madurez porque ayuda con:

- experimentación rápida
- depuración paso a paso
- validación visual de páginas scrapeadas
- reintentos manuales cuando cambia una web
- inspección de datos parcialmente parseados

El inconveniente es que cuesta más automatizarlo de forma robusta y también cuesta más desplegarlo en cloud tal como está.

## 5. Nueva capa de pipeline añadida en el repo

Para pasar de notebooks a automatización, se ha añadido una nueva capa de pipeline:

- `data_pipeline/config.py`
- `data_pipeline/legacy.py`
- `data_pipeline/pipeline.py`
- `data_pipeline/logging_utils.py`
- `scripts/run_pipeline.py`
- `config/scraping_sources.json`

### Qué hace esta nueva capa

- centraliza la configuración de fuentes
- envuelve los módulos legacy de scraping sin reescribirlos todavía
- permite lanzar el flujo desde terminal con un único comando
- prepara el proyecto para automatización local primero y cloud después

### Modos soportados

- `fixtures`
- `download`
- `parse`
- `full`

Ejemplo de comando:

```bash
./venv/bin/python scripts/run_pipeline.py --source whoscored_championship --mode full
```

## 6. Flujo automatizado propuesto a futuro

El flujo recomendado para producción es:

1. se lanza el job programado
2. el pipeline carga la configuración de la fuente
3. se refrescan los fixtures
4. se descargan solo partidos faltantes o incompletos
5. los JSON brutos se parsean a tablas estructuradas
6. los outputs consolidados se validan
7. las tablas finales se cargan en base de datos
8. Streamlit lee los datos actualizados
9. la app muestra la fecha de última actualización correcta

## 7. Integración con Streamlit

La arquitectura limpia es:

- el pipeline de scraping actualiza datos
- Streamlit consume datos

Funcionalidades recomendadas en la app:

- mostrar la última fecha de actualización
- mostrar el estado del pipeline
- opcionalmente incluir un botón admin para lanzar una actualización manual

No recomendado:

- usar Streamlit como scheduler principal

Por qué:

- Streamlit es una capa de interfaz, no una capa robusta de planificación de tareas
- los jobs automáticos deben vivir en un scheduler o proceso de servidor

## 8. Local primero, cloud después

El camino planteado es adecuado y realista:

### Fase local

- correr el pipeline desde un equipo local
- validar fixtures, descargas, outputs parseados y compatibilidad con lo que consumirá la app

### Fase cloud

- mover el mismo pipeline scriptado a una máquina estable en cloud
- configurar dependencias de navegador/headless si Selenium sigue siendo necesario
- programar ejecuciones en horarios fijos
- usar almacenamiento compartido o base de datos compartida con la app

## 9. Riesgos técnicos principales

- las webs fuente pueden cambiar su HTML
- los flujos con Selenium son más frágiles que una API estable
- hay lógica legacy duplicada entre las dos familias de fuentes
- los outputs parseados son útiles pero aún no están completamente consolidados en un único pipeline de producción
- programar ejecuciones en portátiles personales no es estable para producción

## 10. Próximos pasos recomendados

Corto plazo:

- validar el nuevo pipeline scriptado con una ejecución real
- elegir una fuente oficial primaria
- elegir un destino oficial de datos, idealmente MySQL

Medio plazo:

- añadir consolidación y carga a base de datos
- añadir logs de actualización y reporting de errores
- mostrar el estado de actualización dentro de Streamlit

Largo plazo:

- desplegar el pipeline en cloud
- programar ejecuciones automáticas
- reducir la dependencia de notebooks
- dejar los notebooks solo para depuración e investigación

## 11. Resumen simple end-to-end

En una frase:

se descubren los fixtures, se descargan los JSON brutos de los partidos, esos ficheros se transforman en tablas de fútbol estructuradas y esas tablas deberían convertirse en la fuente de datos actualizada regularmente para la aplicación Streamlit.
