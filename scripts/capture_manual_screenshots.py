#!/usr/bin/env python3
"""
Capture end-user manual screenshots for Watford Player Development Hub.

Output:
  manual_assets/screenshots/raw/*.png
  manual_assets/screenshots/manifest.csv
"""

from __future__ import annotations

import csv
import re
import time
import os
from pathlib import Path
from typing import Callable, List, Tuple

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = os.environ.get("MANUAL_BASE_URL", "http://localhost:8501/").rstrip("/") + "/"
OUT_DIR = Path("manual_assets/screenshots/raw")
MANIFEST_PATH = Path("manual_assets/screenshots/manifest.csv")
UPLOAD_FILE = Path("data/Individuals - Training.xlsx").resolve()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def wait_text(driver: webdriver.Chrome, text: str, timeout: int = 45) -> None:
    WebDriverWait(driver, timeout).until(lambda d: text in d.page_source)


def wait_any_text(driver: webdriver.Chrome, texts: List[str], timeout: int = 45) -> str:
    def _has_any(d):
        src = d.page_source
        for t in texts:
            if t in src:
                return t
        return False

    found = WebDriverWait(driver, timeout).until(_has_any)
    return str(found)


def first_displayed(elements):
    for el in elements:
        try:
            if el.is_displayed():
                return el
        except Exception:
            continue
    return None


def safe_click(driver: webdriver.Chrome, xpath: str, timeout: int = 20) -> None:
    el = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
    try:
        WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
        el.click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)


def click_text(driver: webdriver.Chrome, text: str, timeout: int = 20) -> None:
    xpaths = [
        f"//button[normalize-space()='{text}' or .//p[normalize-space()='{text}']]",
        f"//label[.//p[normalize-space()='{text}'] or normalize-space()='{text}']",
        f"//summary[.//p[contains(normalize-space(), '{text}')] or contains(normalize-space(), '{text}')]",
        f"//*[self::p or self::span or self::div][normalize-space()='{text}']",
    ]
    last_exc = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        for xp in xpaths:
            try:
                elems = driver.find_elements(By.XPATH, xp)
                el = first_displayed(elems)
                if not el:
                    continue
                try:
                    el.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", el)
                time.sleep(1.0)
                return
            except Exception as exc:
                last_exc = exc
        time.sleep(0.5)
    raise TimeoutException(f"Unable to click text '{text}'. Last error: {last_exc}")


def set_text_input(driver: webdriver.Chrome, label: str, value: str, index: int = 0, timeout: int = 20) -> None:
    xp = f"//div[@data-testid='stTextInput'][.//p[normalize-space()='{label}']]//input"
    WebDriverWait(driver, timeout).until(EC.presence_of_all_elements_located((By.XPATH, xp)))
    inputs = [e for e in driver.find_elements(By.XPATH, xp) if e.is_displayed()]
    if not inputs:
        raise TimeoutException(f"No visible input for label '{label}'")
    target = inputs[index if index < len(inputs) else 0]
    target.click()
    target.send_keys(Keys.COMMAND, "a")
    target.send_keys(value)
    time.sleep(0.5)


def set_selectbox(driver: webdriver.Chrome, label: str, option_text: str, timeout: int = 20) -> None:
    xp = f"//div[@data-testid='stSelectbox'][.//p[normalize-space()='{label}']]//input[@role='combobox']"
    combo = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xp)))
    combo.click()
    time.sleep(0.3)
    combo.send_keys(Keys.COMMAND, "a")
    combo.send_keys(option_text)
    time.sleep(0.8)
    combo.send_keys(Keys.ENTER)
    time.sleep(1.5)


def open_expander(driver: webdriver.Chrome, title_contains: str, timeout: int = 20) -> None:
    xp = f"//summary[.//p[contains(normalize-space(), '{title_contains}')] or contains(normalize-space(), '{title_contains}')]"
    summary = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xp)))
    try:
        parent = summary.find_element(By.XPATH, "./ancestor::details[1]")
        is_open = parent.get_attribute("open")
    except Exception:
        is_open = None
    if not is_open:
        try:
            summary.click()
        except Exception:
            driver.execute_script("arguments[0].click();", summary)
        time.sleep(1.0)


def save_shot(driver: webdriver.Chrome, shot_id: str, title: str, manifest_rows: List[Tuple[str, str, str]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{shot_id}_{slug(title)}.png"
    path = OUT_DIR / file_name
    driver.save_screenshot(str(path))
    manifest_rows.append((shot_id, title, str(path)))
    print(f"OK {shot_id}: {path}")
    return path


def run() -> int:
    opts = Options()
    opts.binary_location = "/opt/homebrew/bin/chromium"
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=opts)
    wait = WebDriverWait(driver, 45)
    manifest_rows: List[Tuple[str, str, str]] = []
    failures: List[str] = []

    def step(shot_id: str, title: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            time.sleep(1.2)
            save_shot(driver, shot_id, title, manifest_rows)
        except Exception as exc:
            msg = f"{shot_id} - {title}: {exc}"
            print(f"FAIL {msg}")
            failures.append(msg)

    try:
        driver.get(BASE_URL)
        wait_text(driver, "Watford Player Development")
        time.sleep(4.0)

        step("S01", "Login Screen", lambda: None)
        step("S02", "Player Login Fields", lambda: wait_text(driver, "Select Role"))

        step("S03", "Staff Login Fields", lambda: (
            set_selectbox(driver, "Select Role", "Staff"),
            wait_text(driver, "Username"),
        ))

        set_text_input(driver, "Username", "admin")
        set_text_input(driver, "Password", "admin123")
        click_text(driver, "Login")
        wait_text(driver, "Watford Staff Dashboard")
        time.sleep(3)

        step("S04", "Staff Dashboard Navigation", lambda: wait_text(driver, "Staff Info"))

        click_text(driver, "Individual Development")
        wait_text(driver, "Individual Development")
        time.sleep(4)

        step("S05", "Individual Development Sidebar", lambda: wait_text(driver, "General Dashboard"))
        step("S06", "General Dashboard KPIs", lambda: wait_text(driver, "General Dashboard"))
        step("S07", "General Dashboard Chart Summary", lambda: driver.execute_script("window.scrollTo(0, 900);"))

        click_text(driver, "Players Profile")
        wait_text(driver, "Players Profile")
        time.sleep(4)
        driver.execute_script("window.scrollTo(0, 0);")

        step("S08", "Players Profile Selectors", lambda: wait_text(driver, "Select Player"))
        step("S09", "Players Profile Summary", lambda: driver.execute_script("window.scrollTo(0, 650);"))
        step("S10", "Players Profile Activities", lambda: driver.execute_script("window.scrollTo(0, 1200);"))
        step("S11", "Players Profile Timeline", lambda: driver.execute_script("window.scrollTo(0, 1700);"))
        step("S12", "Players Profile PDF Actions", lambda: driver.execute_script("window.scrollTo(0, 2300);"))

        try:
            click_text(driver, "Files")
            wait_text(driver, "Files")
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, 0);")

            step("S13", "Files Current Data Download", lambda: wait_text(driver, "Files"))

            open_expander(driver, "Importar Datos desde Excel")
            step("S14", "Files Upload Validate", lambda: wait_text(driver, "Selecciona un archivo Excel"))

            if UPLOAD_FILE.exists():
                file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
                file_input = first_displayed(file_inputs) or (file_inputs[0] if file_inputs else None)
                if file_input:
                    file_input.send_keys(str(UPLOAD_FILE))
                    time.sleep(1.5)
                    click_text(driver, "🔍 Validar Archivo")
                    wait_text(driver, "Validación exitosa", timeout=90)
                    step("S15", "Files Validation Success", lambda: wait_text(driver, "Resumen de Importación"))
                    step("S16", "Files Confirm Import", lambda: wait_text(driver, "✅ Confirmar Importación"))
                else:
                    failures.append("S15/S16 - File input not found.")
            else:
                failures.append(f"S15/S16 - Upload file not found: {UPLOAD_FILE}")
        except Exception as exc:
            failures.append(f"S13-S16 - Files flow failed: {exc}")

        try:
            driver.get(f"{BASE_URL}player_dashboard")
            wait_any_text(driver, ["Filter Players by Status", "Select Player", "Manage players list"], timeout=90)
            time.sleep(3)

            step("S17", "Player Dashboard Manage Button", lambda: wait_any_text(driver, ["Manage players list", "Select Player"], timeout=60))

            # Select a player to fully render dashboard content.
            set_selectbox(driver, "Select Player", "Daniel Bachmann")
            time.sleep(4)
            wait_text(driver, "Daniel Bachmann")

            step("S22", "Player Dashboard Season Time Filters", lambda: wait_text(driver, "Season & Time Filters"))
            step("S23", "Overview Match Filter", lambda: wait_text(driver, "Filter by Match (click to hide)"))
            step("S24", "Overview KPI Cards", lambda: driver.execute_script("window.scrollTo(0, 700);"))
            step("S27", "Floating Download PDF", lambda: driver.execute_script("window.scrollTo(0, 0);"))
            step("S28", "Sidebar Logout", lambda: wait_text(driver, "Logout"))

            click_text(driver, "⚙️ Manage players list")
            wait_any_text(driver, ["Gestionar lista de jugadores", "Manage players list"], timeout=60)
            time.sleep(3)

            open_expander(driver, "Añadir jugador")
            step("S18", "Manage Players Add Lookup", lambda: wait_text(driver, "Nombre del jugador"))
            step("S19", "Manage Players Card Actions", lambda: driver.execute_script("window.scrollTo(0, 850);"))

            # Open first info button.
            info_buttons = driver.find_elements(By.XPATH, "//button[normalize-space()='ℹ️' or .//p[normalize-space()='ℹ️']]")
            info_btn = first_displayed(info_buttons)
            if info_btn:
                try:
                    info_btn.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", info_btn)
                time.sleep(2)
                step("S20", "Manage Players Edit Details", lambda: wait_text(driver, "Detalles editables"))
            else:
                failures.append("S20 - Info button not found.")

            open_expander(driver, "Vista avanzada (tabla)")
            step("S21", "Manage Players Advanced Table", lambda: wait_text(driver, "Guardar cambios (tabla)"))

            driver.get(f"{BASE_URL}player_dashboard")
            wait_text(driver, "Daniel Bachmann", timeout=90)
            time.sleep(4)
            click_text(driver, "Trends Stats")
            wait_text(driver, "Performance Trends Over Time")
            step("S25", "Trends Stats Charts", lambda: wait_text(driver, "Filter by Match (click to hide)"))

            click_text(driver, "Player Comparison")
            wait_text(driver, "Top Players in the Competition", timeout=120)
            open_expander(driver, "Filter by Teams")
            open_expander(driver, "Filter Players by Position")
            step("S26", "Player Comparison Filters Charts", lambda: wait_text(driver, "Comparison by KPI", timeout=90))
        except Exception as exc:
            failures.append(f"S17-S28 - Player/Manage flow failed: {exc}")

    finally:
        driver.quit()

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["screenshot_id", "title", "path"])
        writer.writerows(manifest_rows)

    print("")
    print(f"Captured: {len(manifest_rows)} screenshots")
    print(f"Manifest: {MANIFEST_PATH}")
    if failures:
        print("Failures:")
        for item in failures:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
