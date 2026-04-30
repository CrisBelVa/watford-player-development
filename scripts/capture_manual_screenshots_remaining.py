#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = os.environ.get("MANUAL_BASE_URL", "http://localhost:8502/").rstrip("/") + "/"
OUT_DIR = Path("manual_assets/screenshots/raw")
UPLOAD_FILE = Path("data/Individuals - Training.xlsx").resolve()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save(driver: webdriver.Chrome, shot_id: str, title: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{shot_id}_{slug(title)}.png"
    driver.save_screenshot(str(path))
    print(f"OK {shot_id}: {path}")


def wait_has(driver: webdriver.Chrome, text: str, timeout: int = 45) -> None:
    WebDriverWait(driver, timeout).until(lambda d: text in d.page_source)


def wait_any(driver: webdriver.Chrome, texts: list[str], timeout: int = 45) -> str:
    def cond(d):
        src = d.page_source
        for t in texts:
            if t in src:
                return t
        return False

    return WebDriverWait(driver, timeout).until(cond)


def assert_logged(driver: webdriver.Chrome) -> None:
    src = driver.page_source
    if "You must be logged in to view this page." in src or "You must be logged in as staff to view this page." in src:
        raise RuntimeError("Authentication lost in current page.")


def click_xpath(driver: webdriver.Chrome, xpath: str, timeout: int = 25) -> None:
    end = time.time() + timeout
    last_err = None
    while time.time() < end:
        try:
            elems = driver.find_elements(By.XPATH, xpath)
            for el in elems:
                if not el.is_displayed():
                    continue
                try:
                    el.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", el)
                time.sleep(1.2)
                return
        except Exception as exc:
            last_err = exc
        time.sleep(0.4)
    raise TimeoutException(f"Could not click xpath: {xpath}. Last error: {last_err}")


def click_label(driver: webdriver.Chrome, text: str) -> None:
    click_xpath(driver, f"//label[.//p[normalize-space()='{text}'] or normalize-space()='{text}']")


def click_button_text_contains(driver: webdriver.Chrome, token: str) -> None:
    token_l = token.lower()
    end = time.time() + 20
    while time.time() < end:
        buttons = driver.find_elements(By.XPATH, "//button")
        for b in buttons:
            try:
                if not b.is_displayed():
                    continue
                txt = (b.text or "").strip().lower()
                if token_l in txt:
                    try:
                        b.click()
                    except Exception:
                        driver.execute_script("arguments[0].click();", b)
                    time.sleep(1.2)
                    return
            except Exception:
                continue
        time.sleep(0.3)
    raise TimeoutException(f"Button containing '{token}' not found")


def set_selectbox(driver: webdriver.Chrome, label: str, option: str) -> None:
    xp = f"//div[@data-testid='stSelectbox'][.//p[normalize-space()='{label}']]//input[@role='combobox']"
    inp = WebDriverWait(driver, 30).until(lambda d: d.find_element(By.XPATH, xp))
    inp.click()
    inp.send_keys(Keys.COMMAND, "a")
    inp.send_keys(option)
    time.sleep(1)
    inp.send_keys(Keys.ENTER)
    time.sleep(2)


def set_text_input(driver: webdriver.Chrome, label: str, value: str) -> None:
    xp = f"//div[@data-testid='stTextInput'][.//p[normalize-space()='{label}']]//input"
    inputs = [e for e in driver.find_elements(By.XPATH, xp) if e.is_displayed()]
    if not inputs:
        raise TimeoutException(f"Input '{label}' not found")
    i = inputs[0]
    i.click()
    i.send_keys(Keys.COMMAND, "a")
    i.send_keys(value)
    time.sleep(0.3)


def open_sidebar_link(driver: webdriver.Chrome, route: str) -> None:
    href = f"{BASE_URL}{route.lstrip('/')}"
    click_xpath(driver, f"//a[@data-testid='stSidebarNavLink' and @href='{href}']")
    time.sleep(5)


def run() -> int:
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
        # Login Staff
        driver.get(BASE_URL)
        time.sleep(4)
        set_selectbox(driver, "Select Role", "Staff")
        set_text_input(driver, "Username", "admin")
        set_text_input(driver, "Password", "admin123")
        click_xpath(driver, "//button[.//p[normalize-space()='Login'] or normalize-space()='Login']")
        wait_has(driver, "Watford Staff Dashboard", timeout=45)
        assert_logged(driver)

        # Staff -> Individual Development
        click_label(driver, "Individual Development")
        wait_any(driver, ["Individual Development", "General Dashboard"], timeout=45)
        assert_logged(driver)

        # Files workflow
        click_label(driver, "Files")
        wait_any(driver, ["Files", "Import Data from Excel", "Importar Datos desde Excel"], timeout=45)
        assert_logged(driver)

        # Open import expander (EN or ES)
        try:
            click_xpath(driver, "//summary[contains(., 'Import') and contains(., 'Excel')]")
        except Exception:
            click_xpath(driver, "//summary[contains(., 'Importar') and contains(., 'Excel')]")
        time.sleep(1.5)
        save(driver, "S14", "Files Upload Validate")

        # Upload file and validate
        try:
            files = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            target = None
            for f in files:
                if f.is_displayed():
                    target = f
                    break
            if target is None and files:
                target = files[0]
            if target and UPLOAD_FILE.exists():
                target.send_keys(str(UPLOAD_FILE))
                time.sleep(2.5)
                save(driver, "S15", "Files Validation Success")
                try:
                    click_button_text_contains(driver, "valid")
                except Exception:
                    click_button_text_contains(driver, "validar")
                time.sleep(5)
                save(driver, "S16", "Files Confirm Import")
            else:
                failures.append("S15/S16: file uploader unavailable.")
        except Exception as exc:
            failures.append(f"S15/S16 failed: {exc}")

        # Player dashboard from sidebar nav
        open_sidebar_link(driver, "player_dashboard")
        wait_any(driver, ["Filter Players by Status", "Manage players list", "Select Player"], timeout=60)
        assert_logged(driver)
        save(driver, "S17", "Player Dashboard Manage Button")

        # Select a player and capture dashboard
        try:
            set_selectbox(driver, "Select Player", "Daniel Bachmann")
            wait_any(driver, ["Overview Stats", "Filter by Match (click to hide)"], timeout=60)
        except Exception as exc:
            failures.append(f"Selecting player failed: {exc}")
        assert_logged(driver)
        save(driver, "S22", "Player Dashboard Season Time Filters")
        save(driver, "S23", "Overview Match Filter")
        driver.execute_script("window.scrollTo(0, 700);")
        time.sleep(1.2)
        save(driver, "S24", "Overview KPI Cards")
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1.0)
        save(driver, "S27", "Floating Download PDF")
        save(driver, "S28", "Sidebar Logout")

        # Manage players
        open_sidebar_link(driver, "manage_players")
        wait_any(driver, ["Manage players list", "Gestionar lista de jugadores"], timeout=60)
        assert_logged(driver)
        click_xpath(driver, "//summary[contains(., 'Añadir jugador') or contains(., 'Add player')]")
        time.sleep(1.2)
        save(driver, "S18", "Manage Players Add Lookup")
        driver.execute_script("window.scrollTo(0, 900);")
        time.sleep(1.2)
        save(driver, "S19", "Manage Players Card Actions")
        try:
            click_xpath(driver, "//button[normalize-space()='ℹ️' or .//p[normalize-space()='ℹ️']]", timeout=8)
        except Exception:
            pass
        time.sleep(1.2)
        save(driver, "S20", "Manage Players Edit Details")
        try:
            click_xpath(driver, "//summary[contains(., 'Vista avanzada') or contains(., 'Advanced view')]")
        except Exception:
            pass
        time.sleep(1.2)
        save(driver, "S21", "Manage Players Advanced Table")

        # Trends and Player Comparison
        open_sidebar_link(driver, "player_dashboard")
        wait_any(driver, ["Overview Stats", "Filter by Match (click to hide)"], timeout=60)
        assert_logged(driver)
        click_label(driver, "Trends Stats")
        wait_has(driver, "Performance Trends Over Time", timeout=60)
        save(driver, "S25", "Trends Stats Charts")

        click_label(driver, "Player Comparison")
        wait_any(driver, ["Top Players in the Competition", "Comparison by KPI"], timeout=120)
        try:
            click_xpath(driver, "//summary[contains(., 'Filter by Teams')]")
        except Exception:
            pass
        try:
            click_xpath(driver, "//summary[contains(., 'Filter Players by Position')]")
        except Exception:
            pass
        save(driver, "S26", "Player Comparison Filters Charts")

    finally:
        driver.quit()

    if failures:
        print("Failures:")
        for f in failures:
            print("-", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
