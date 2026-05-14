# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 23:58:21 2025

@author: aleex
"""

import os
import shutil
import subprocess
from datetime import datetime
import glob
import pandas as pd  # Importa la librería pandas y la asigna al alias 'pd', utilizada para la manipulación y análisis de datos.
import time  # Importa el módulo time, que proporciona diversas funciones relacionadas con la manipulación del tiempo, como delays y medición de intervalos.
import json
import numpy as np
import random
import time
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from pandas import json_normalize
from selenium import webdriver  # Importa el módulo webdriver de Selenium, utilizado para automatizar navegadores web.
from selenium.webdriver.common.by import By  # Importa el módulo By, que proporciona métodos para localizar elementos en una página web por diferentes atributos (ID, nombre, clase, etc.).
from selenium.webdriver.support import expected_conditions as EC  # Importa el módulo expected_conditions, que contiene una serie de condiciones que se pueden esperar (esperar a que un elemento sea clicable, visible, etc.).
from selenium.webdriver.support.ui import WebDriverWait  # Importa el módulo WebDriverWait, que permite esperar hasta que se cumpla una condición específica.
from selenium.webdriver.chrome.service import Service  # Importa el módulo Service, que permite iniciar y controlar el servicio del navegador Chrome.
from selenium.webdriver.chrome.options import Options  # Importa el módulo Options, que permite personalizar las opciones de configuración del navegador Chrome.
from webdriver_manager.chrome import ChromeDriverManager  # Importa el ChromeDriverManager, que gestiona automáticamente la descarga e instalación del controlador de Chrome.
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    NoSuchElementException,
    ElementClickInterceptedException,
)
# Configuración
import warnings  # Importa el módulo warnings, que maneja los mensajes de advertencia en Python.
warnings.filterwarnings("ignore")  # Configura para ignorar todas las advertencias.



def random_sleep_time():
    return random.uniform(3, 6)

def _read_major_version(binary_path, version_arg="--version"):
    if not binary_path or not os.path.exists(binary_path):
        return None
    try:
        result = subprocess.run(
            [binary_path, version_arg],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    text = f"{result.stdout} {result.stderr}".strip()
    match = re.search(r"(\d+)\.", text)
    return int(match.group(1)) if match else None

def setup_driver():
    options = Options()
    is_headless = os.getenv("HEADLESS", "").lower() in {"1", "true", "yes"} or os.getenv("CI", "").lower() == "true"
    chrome_binary = (
        os.getenv("GOOGLE_CHROME_BIN")
        or os.getenv("CHROME_BINARY")
        or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH") or shutil.which("chromedriver")

    if is_headless:
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
    else:
        options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")

    if chrome_binary and os.path.exists(chrome_binary):
        options.binary_location = chrome_binary

    chrome_major = _read_major_version(chrome_binary) if chrome_binary else None
    driver_major = _read_major_version(chromedriver_path) if chromedriver_path else None

    service = None
    if chromedriver_path and os.path.exists(chromedriver_path):
        if chrome_major and driver_major and chrome_major == driver_major:
            service = Service(chromedriver_path)
        else:
            print(
                f"Chromedriver local omitido por incompatibilidad "
                f"(driver={driver_major}, chrome={chrome_major})."
            )

    if service is None:
        try:
            driver = webdriver.Chrome(options=options)
            return driver
        except Exception as first_error:
            print(f"Selenium Manager no pudo iniciar Chrome automáticamente: {first_error}")
            service = Service(ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=options)
    return driver


def dismiss_blocking_overlays(driver):
    """Try to remove or hide common floating layers that block clicks."""
    selectors = [
        ".webpush-swal2-container",
        ".qc-cmp2-container",
        "#qc-cmp2-container",
        "[id*='onetrust']",
        "[class*='onetrust']",
        "iframe[id*='sp_message']",
        "iframe[src*='consent']",
        "div[style*='z-index: 2147483647']",
    ]

    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                if element.is_displayed():
                    driver.execute_script(
                        """
                        arguments[0].style.display = 'none';
                        arguments[0].style.visibility = 'hidden';
                        arguments[0].remove();
                        """,
                        element,
                    )
        except Exception:
            pass


def safe_click(driver, locator, timeout=10):
    """Scroll into view and click, falling back to JavaScript if needed."""
    element = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(locator)
    )
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
        element,
    )
    time.sleep(1)

    try:
        WebDriverWait(driver, timeout).until(EC.element_to_be_clickable(locator))
        element.click()
        return element
    except (ElementClickInterceptedException, TimeoutException):
        dismiss_blocking_overlays(driver)
        try:
            close_ad_popup(driver)
        except Exception:
            pass
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(locator)
        )
        driver.execute_script("arguments[0].click();", element)
        return element


def optional_click(driver, locators, timeout=10):
    """Try several locators and continue if none works."""
    for locator in locators:
        try:
            return safe_click(driver, locator, timeout=timeout)
        except Exception:
            continue
    return None


def optional_text(driver, locators, timeout=10):
    """Return text from the first locator that resolves, else None."""
    for locator in locators:
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            text = element.text.strip()
            if text:
                return text
        except Exception:
            continue
    return None

def close_ad_popup(driver):
    try:
        # Esperar hasta 10 segundos para que aparezca la publicidad
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "webpush-swal2-container"))
        )
        
        # Intentar cerrar con el botón 'x'
        try:
            close_button = driver.find_element(By.CLASS_NAME, "webpush-swal2-close")
            close_button.click()
            print("Publicidad cerrada exitosamente con el botón 'x'.")
            return
        except NoSuchElementException:
            print("No se encontró el botón 'x' para cerrar la publicidad.")
        
        # Si no se pudo cerrar con 'x', intentar con el botón "FREE TRIAL"
        try:
            free_trial_button = driver.find_element(By.XPATH, "//button[contains(@class, 'webpush-swal2-confirm')]")
            free_trial_button.click()
            print("Publicidad cerrada exitosamente con el botón 'FREE TRIAL'.")
            return
        except NoSuchElementException:
            print("No se encontró el botón 'FREE TRIAL'.")
        
        # Si aún no se ha cerrado, intentar hacer clic fuera de la publicidad
        try:
            overlay = driver.find_element(By.CLASS_NAME, "webpush-swal2-container")
            driver.execute_script("arguments[0].click();", overlay)
            print("Se intentó cerrar la publicidad haciendo clic fuera de ella.")
        except Exception as e:
            print(f"No se pudo cerrar la publicidad haciendo clic fuera: {str(e)}")
        
    except TimeoutException:
        print("No se encontró la publicidad o no apareció en el tiempo esperado.")
    except Exception as e:
        print(f"Error inesperado al intentar cerrar la publicidad: {str(e)}")

def extract_matches_from_html(html_content, fixtures_url):
    soup = BeautifulSoup(html_content, 'html.parser')
    matches = []
    
    date_accordions = soup.find_all('div', class_='Accordion-module_accordion__UuHD0')
    
    from urllib.parse import urlparse

    
    # Parseamos la URL para obtener la ruta
    parsed_url = urlparse(fixtures_url)
    # Eliminamos posibles barras al inicio/final y separamos por "/"
    segments = parsed_url.path.strip("/").split("/")

    region=""
    tournament=""
    stage=""
    competition=""
    season=[]
    try:
        region = segments[1]
        tournament = segments[3]
        stage = segments[5]
        competition = segments[-1]
    except:
        pass
        
        # print("Region:", region)
        # print("Tournament:", tournament)
        # print("Season:", season)
        # print("Stage:", stage)
        # print("Competition:", competition)
    else:
        # print("La URL no tiene el formato esperado.")
        pass

    for accordion in date_accordions:
        date = accordion.find('div', class_='Accordion-module_header__HqzWD').text.strip()
        match_divs = accordion.find_all('div', class_='Match-module_match__XlKTY')
        print("Match:", match_divs)
        for match in match_divs:
            try:
                time_or_status = match.find('span', class_=['Match-module_startTime__c49c8', 'Match-module_FT__2rmH7'])
                time_or_status = time_or_status.text.strip() if time_or_status else "N/A"
                
                teams = match.find_all('a', class_='Match-module_teamNameText__Dqv-G')
                home_team = teams[0].text.strip() if teams else "N/A"
                away_team = teams[1].text.strip() if len(teams) > 1 else "N/A"
                
                # score_element = match.find('a', class_='Match-module_score__5Ghhj')
                # match_id = score_element['id'].split('-')[1] if score_element else "N/A"
                score_elements = match.find_all('a', id=re.compile('^scoresBtn-'))
                # Extraer el ID del partido del atributo id
                score_element = score_elements[0]
                match_id = score_element['id'].split('-')[1]
                
                scores = score_element.find_all('span') if score_element else []
                home_score = scores[0].text.strip() if scores else "N/A"
                away_score = scores[1].text.strip() if len(scores) > 1 else "N/A"
                
                odds = match.find_all('span', class_='OddsButton-module_oddsText__WD5Dv')
                home_odds = odds[0].text.strip() if odds else "N/A"
                draw_odds = odds[1].text.strip() if len(odds) > 1 else "N/A"
                away_odds = odds[2].text.strip() if len(odds) > 2 else "N/A"
                
                matches.append({
                    'region': region,
                    'tournament':tournament,
                    'season':season,
                    'stage':stage,
                    'competition':competition,
                    'date': date,
                    'time_or_status': time_or_status,
                    'home_team': home_team,
                    'away_team': away_team,
                    'match_id': match_id,
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds
                })
            except Exception as e:
                print(f"Error procesando un partido: {str(e)}")
    
    return pd.DataFrame(matches)

def scrape_fixtures(fixtures_url, mes_ini=200001):
    driver = setup_driver()
    #fixtures_url = "https://1xbet.whoscored.com/regions/252/tournaments/7/england-championship"
    driver.get(fixtures_url)
    inicio = datetime.strptime(str(mes_ini), "%Y%m")
    all_matches_df = pd.DataFrame()

    try:
        print("Esperando que la página cargue completamente...")
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        dismiss_blocking_overlays(driver)
        close_ad_popup(driver)

        # --- Retrocedemos hasta el inicio del calendario usando el botón "previo" ---
        span_locators = [
            (By.CSS_SELECTOR, "span.toggleDatePicker"),
            (By.CSS_SELECTOR, "[class*='toggleDatePicker']"),
            (By.CSS_SELECTOR, "[class*='Calendar-module_header']"),
        ]
        prev_button_locator = (By.ID, "dayChangeBtn-prev")
        optional_click(
            driver,
            [
                *span_locators,
                (By.CSS_SELECTOR, "[class*='toggleDatePicker']"),
                (By.CSS_SELECTOR, "[class*='Calendar-module_dayChangeBtn']"),
            ],
            timeout=20,
        )
        # Obtenemos el texto actual del span (mes actual)
        current_month = optional_text(driver, span_locators, timeout=10)
        if current_month:
            print("Mes actual (inicio antes de retroceder):", current_month)
        else:
            print("No se pudo leer el encabezado del calendario; continuamos con la vista actual.")

        # Hacemos clic en el botón "previo" hasta que el mes ya no cambie
        if current_month:
            while True:
                try:
                    dismiss_blocking_overlays(driver)
                    safe_click(driver, prev_button_locator, timeout=15)
                    time.sleep(2)  # Pequeña espera para la animación
                except Exception as e:
                    print("Error al hacer clic en el botón 'previo':", e)
                    break

                new_month = optional_text(driver, span_locators, timeout=10)
                if not new_month:
                    print("No se pudo leer el nuevo encabezado del calendario tras retroceder.")
                    break
                new_month_ini = "{} {} {}".format(1,
                                                  new_month.split(" ")[-2].strip(),
                                                  new_month.split(" ")[-1].strip())
                print("Nuevo mes tras clic en 'previo':", new_month)
                if new_month == current_month or datetime.strptime(new_month_ini, "%d %b %Y") <= inicio:
                    print("Ya no se puede retroceder más. Se alcanzó el inicio del calendario.")
                    break
                else:
                    current_month = new_month

        # --- Ahora avanzamos mes a mes usando el botón "next" ---
        month_index = 1  # para el mensaje de procesamiento
        next_button_locator = (By.ID, "dayChangeBtn-next")
        # Obtenemos el mes actual (después de haber retrocedido)
        current_month = optional_text(driver, span_locators, timeout=10)
        if current_month:
            print("Mes de inicio para avanzar:", current_month)

        while True:
            print(f"Intentando procesar el mes {month_index}")
            
            # Esperar a que la página se actualice con el contenido del mes seleccionado
            time.sleep(5)

            print(f"Extrayendo datos para el mes {month_index}")
            page_source = driver.page_source
            try:
                matches_df = extract_matches_from_html(page_source, fixtures_url)
            except Exception as e:
                print(f"Error extrayendo datos para el mes {month_index}: {e}")
                matches_df = pd.DataFrame()

            all_matches_df = pd.concat([all_matches_df, matches_df], ignore_index=True)
            print(f"Datos extraídos para el mes {month_index}. Partidos en este mes: {len(matches_df)}")

            # Hacemos clic en el botón "next" para avanzar
            try:
                dismiss_blocking_overlays(driver)
                safe_click(driver, next_button_locator, timeout=15)
            except Exception as e:
                print("Error al hacer clic en el botón 'next':", e)
                break

            # Esperamos a que el span del mes se actualice y lo obtenemos
            try:
                new_month = optional_text(driver, span_locators, timeout=10)
            except Exception as e:
                print("Error al obtener el nuevo mes:", e)
                break
            if not new_month:
                print("No se pudo leer el encabezado tras avanzar; detenemos la iteración.")
                break
            #new_month_ini = new_month.split(" - ")[-1].strip()
            # Si el mes no cambia, se termina el bucle
            if new_month == current_month:
                #or datetime.strptime(new_month_ini, "%d %b %Y") > datetime.today()
                print(f"El mes no cambió tras hacer clic en 'next'. Se ha llegado al final: {new_month}")
                break
            else:
                print(f"Avanzamos al mes: {new_month}")
                current_month = new_month
                month_index += 1
    except Exception as e:
        print(f"Se produjo un error general: {str(e)}")
    finally:
        driver.quit()

    if all_matches_df.empty:
        raise RuntimeError(
            "No fixtures were extracted from WhoScored. The page structure or a blocking overlay likely changed."
        )

    return all_matches_df

def to_int(val):
    """ Reconoce valores numericos y los transforma a enteros.
    """
    try:
        value = int(float(val))
    except ValueError:
        value = ""
    return value

def transform_date(date_str):
    # date_obj = datetime.strptime(date_str, "%A, %b %d %Y")
    # return date_obj.strftime("%Y%m%d")
    cleaned_date = re.search(r'\b\w{3} \d{2} \d{4}\b', date_str).group()
    print(cleaned_date)
    return cleaned_date

def get_json_games(df,ruta,ini=200001):
    df['new_date'] = pd.to_datetime(df['date'], format="%A, %b %d %Y")
    df=df[df['new_date'] <= datetime.today()]
    df["match_id"] = df["match_id"].map(to_int)
    driver = setup_driver()
    
    if ini:
        df=df[df['new_date'] >= datetime.strptime(str(ini), "%Y%m")]

    # Contador de exportaciones
    total_exportaciones = 0
    
    # Iterar sobre cada fila del DataFrame
    for index, row in df.iterrows():
        match_id = row['match_id']
        
        # SOLO DESCARGAR SI NO EXISTE EL ARCHIVO
        PARTIDOS_DIR = ruta
        # Usamos glob para buscar archivos que terminen en _<match_id>.json
        pattern = os.path.join(PARTIDOS_DIR, f"**/*_{match_id}.json")
        files = glob.glob(pattern, recursive=True)
    
        descargar='si'
        if not files: # no existe el archivo lo descargo
            descargar='si'
        else: # existe el a pero el status code debe ser diferente de 6 para descargarse
            
            file_path = files[0]
            print(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    match_data = json.load(f)
                
                #  not files
                # Extraemos la información general que se encuentra en el nodo matchCentreData
                match_centre = match_data.get("matchCentreData", {})
                # print(file_path)
                #print(match_centre.get("statusCode"))
                if match_centre:
                    if match_centre.get("statusCode")==6:
                        descargar='no'
            except:
                pass
            # print(type(match_centre.get("statusCode")))
            
        if descargar=='si':
    
            url = f'https://es.whoscored.com/Matches/{match_id}/Live/'
            
            print(f"Procesando partido: {row['home_team']} vs {row['away_team']} (ID: {match_id})")
            
            try:
                # Acceder a la página
                driver.get(url)
                dismiss_blocking_overlays(driver)
                close_ad_popup(driver)
                
                # Esperar un poco para que se cargue el contenido dinámico
                time.sleep(5)
                
                # Obtener el contenido de la página
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # Buscar el script que contiene los datos
                scripts = soup.find_all("script")
                datos = ''
                for script in scripts:
                    if script.string and "matchCentreData" in script.string:
                        datos = script.string
                        break
                
                if datos:
                    # Extraer los datos JSON
                    listVar = datos.split(';')
                    matchCenterData = listVar[0].split('require.config.params["args"] = ')[1]
                    matchCenterData = matchCenterData.replace('\n','').replace('matchId','"matchId"')
                    matchCenterData = matchCenterData.replace('matchCentreData','"matchCentreData"')
                    matchCenterData = matchCenterData.replace('formationIdNameMappings','"formationIdNameMappings"')
                    matchCenterData = matchCenterData.replace('matchCentreEventTypeJson','"matchCentreEventTypeJson"')
                    
                    # Cargar los datos JSON
                    data = json.loads(matchCenterData)
                    
                    # Crear el nombre del archivo
                    file_name = f"{transform_date(row['date'])}_{row['home_team']}_{row['away_team']}_{row['match_id']}.json"
                    
                    # Guardar el JSON
                    with open(os.path.join(ruta, file_name), 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    
                    print(f"Archivo exportado: {file_name}")
                    total_exportaciones += 1
                else:
                    print(f"No se encontraron datos para el partido {match_id}")
            except Exception as e:
                print(f"Error al procesar el partido {match_id}: {str(e)}")
    
    # Cerrar el driver
    driver.quit()
    
    # Imprimir el total de exportaciones
    print(f"\nTotal de partidos exportados: {total_exportaciones}")
