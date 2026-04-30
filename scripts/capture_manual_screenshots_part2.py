#!/usr/bin/env python3
"""
Capture remaining screenshots (S14-S28) using in-app navigation links to preserve session.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Iterable

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

BASE_URL = os.environ.get("MANUAL_BASE_URL", "http://localhost:8502/").rstrip("/") + "/"
OUT_DIR = Path("manual_assets/screenshots/raw")
UPLOAD_FILE = Path("data/Individuals - Training.xlsx").resolve()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def shot(driver: webdriver.Chrome, shot_id: str, title: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{shot_id}_{slug(title)}.png"
    driver.save_screenshot(str(path))
    print(f"OK {shot_id}: {path}")


def click_any_text(driver: webdriver.Chrome, labels: Iterable[str], timeout: int = 20) -> bool:
    end = time.time() + timeout
    labels_l = [x.lower() for x in labels]
    while time.time() < end:
        # Most reliable first: buttons/labels/summaries/links
        candidates = driver.find_elements(
            By.XPATH, "//button|//label|//summary|//a|//p|//span|//div"
        )
        for el in candidates:
            try:
                if not el.is_displayed():
                    continue
                txt = (el.text or "").strip()
                if not txt:
                    continue
                txt_l = txt.lower()
                if any(lbl in txt_l for lbl in labels_l):
                    try:
                        el.click()
                    except Exception:
                        driver.execute_script("arguments[0].click();", el)
                    time.sleep(1.2)
                    return True
            except Exception:
                continue
        time.sleep(0.4)
    return False


def set_selectbox(driver: webdriver.Chrome, label: str, option: str) -> None:
    xp = f"//div[@data-testid='stSelectbox'][.//p[normalize-space()='{label}']]//input[@role='combobox']"
    combo = driver.find_element(By.XPATH, xp)
    combo.click()
    combo.send_keys(Keys.COMMAND, "a")
    combo.send_keys(option)
    time.sleep(0.8)
    combo.send_keys(Keys.ENTER)
    time.sleep(2.0)


def set_input(driver: webdriver.Chrome, label: str, value: str) -> None:
    xp = f"//div[@data-testid='stTextInput'][.//p[normalize-space()='{label}']]//input"
    inputs = [e for e in driver.find_elements(By.XPATH, xp) if e.is_displayed()]
    if not inputs:
        raise NoSuchElementException(f"Input '{label}' not found")
    inp = inputs[0]
    inp.click()
    inp.send_keys(Keys.COMMAND, "a")
    inp.send_keys(value)
    time.sleep(0.3)


def open_sidebar_link(driver: webdriver.Chrome, route: str) -> None:
    href = f"{BASE_URL}{route.lstrip('/')}"
    link = driver.find_element(By.CSS_SELECTOR, f"a[href='{href}']")
    try:
        link.click()
    except Exception:
        driver.execute_script("arguments[0].click();", link)
    time.sleep(6)


def main() -> int:
    opts = Options()
    opts.binary_location = "/opt/homebrew/bin/chromium"
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=opts)

    failures: list[str] = []

    try:
        # Login as staff.
        driver.get(BASE_URL)
        time.sleep(4)
        set_selectbox(driver, "Select Role", "Staff")
        set_input(driver, "Username", "admin")
        set_input(driver, "Password", "admin123")
        click_any_text(driver, ["login"])
        time.sleep(4)

        # Staff dashboard -> Individual Development.
        click_any_text(driver, ["individual development"])
        time.sleep(6)

        # Files tab and import screens.
        if not click_any_text(driver, ["files"], timeout=12):
            failures.append("Could not click Files tab.")
        time.sleep(4)

        # Open import expander regardless of language.
        if not click_any_text(driver, ["import data from excel", "importar datos desde excel", "import"], timeout=12):
            failures.append("Could not open import expander.")
        time.sleep(2)
        shot(driver, "S14", "Files Upload Validate")

        try:
            file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            file_input = None
            for f in file_inputs:
                if f.is_displayed():
                    file_input = f
                    break
            if file_input is None and file_inputs:
                file_input = file_inputs[0]
            if file_input and UPLOAD_FILE.exists():
                file_input.send_keys(str(UPLOAD_FILE))
                time.sleep(2)
                shot(driver, "S15", "Files Validation Success")
                click_any_text(driver, ["validar", "validate"], timeout=10)
                time.sleep(4)
                shot(driver, "S16", "Files Confirm Import")
            else:
                failures.append("File uploader or upload file not available for S15/S16.")
        except Exception as exc:
            failures.append(f"S15/S16 failed: {exc}")

        # Navigate via sidebar link to player dashboard to preserve session.
        open_sidebar_link(driver, "player_dashboard")
        shot(driver, "S17", "Player Dashboard Manage Button")

        # Select player and capture core dashboard.
        try:
            set_selectbox(driver, "Select Player", "Daniel Bachmann")
            time.sleep(6)
        except Exception as exc:
            failures.append(f"Could not select player: {exc}")

        shot(driver, "S22", "Player Dashboard Season Time Filters")
        shot(driver, "S23", "Overview Match Filter")
        driver.execute_script("window.scrollTo(0, 700);")
        time.sleep(1.5)
        shot(driver, "S24", "Overview KPI Cards")
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1.0)
        shot(driver, "S27", "Floating Download PDF")
        shot(driver, "S28", "Sidebar Logout")

        # Manage players page from sidebar nav.
        open_sidebar_link(driver, "manage_players")
        time.sleep(3)
        click_any_text(driver, ["añadir jugador", "add player"], timeout=8)
        time.sleep(1.5)
        shot(driver, "S18", "Manage Players Add Lookup")

        driver.execute_script("window.scrollTo(0, 900);")
        time.sleep(1.5)
        shot(driver, "S19", "Manage Players Card Actions")

        click_any_text(driver, ["ℹ️"], timeout=8)
        time.sleep(2)
        shot(driver, "S20", "Manage Players Edit Details")

        click_any_text(driver, ["vista avanzada", "advanced view"], timeout=10)
        time.sleep(1.5)
        shot(driver, "S21", "Manage Players Advanced Table")

        # Back to player dashboard for Trends and Comparison.
        open_sidebar_link(driver, "player_dashboard")
        time.sleep(4)
        click_any_text(driver, ["trends stats"], timeout=10)
        time.sleep(4)
        shot(driver, "S25", "Trends Stats Charts")

        click_any_text(driver, ["player comparison"], timeout=10)
        time.sleep(8)
        click_any_text(driver, ["filter by teams"], timeout=8)
        click_any_text(driver, ["filter players by position"], timeout=8)
        shot(driver, "S26", "Player Comparison Filters Charts")

    finally:
        driver.quit()

    if failures:
        print("Failures:")
        for f in failures:
            print("-", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
