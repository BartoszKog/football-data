import subprocess
import time
from pathlib import Path
import sys

# --- KONFIGURACJA ---
# 1. Wpisz sezon, do którego mają trafić mecze
SEASON = "current"

# 2. Wklej tutaj linki do meczów, które chcesz pobrać ręcznie (w cudzysłowach, oddzielone przecinkami)
MANUAL_URLS = [
    "https://www.oddsportal.com/football/poland/division-1/slask-wroclaw-legnica-UVGFwu4c/"
    #"https://www.oddsportal.com/football/poland/ekstraklasa/rakow-czestochowa-lech-poznan-W4aM1b23/",
    # "https://www.oddsportal.com/kolejny-mecz...",
]

LEAGUE = "poland-1-liga"
MARKETS = "1x2,btts,over_under_2_5"

# --- USTAWIENIA ŚCIEŻEK ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Folder football-data/
DATA_DIR = BASE_DIR / "data" / "raw"
LOGS_DIR = BASE_DIR / "logs"

# Ścieżka do Pythona wewnątrz OddsHarvester
SCRAPER_DIR = BASE_DIR / "OddsHarvester"
SCRAPER_PYTHON = SCRAPER_DIR / ".venv" / "Scripts" / "python.exe"

# prefix nazwy pliku na nazwę ligii
LEAGUE_PREFIX = "1liga300326current_"

# --------------------

def run_manual_retry():
    # Sprawdzenie środowiska
    if not SCRAPER_PYTHON.exists():
        print(f"❌ BŁĄD: Nie znaleziono pythona w {SCRAPER_PYTHON}")
        return

    if not MANUAL_URLS:
        print("⚠️ Lista MANUAL_URLS jest pusta. Uzupełnij linki w skrypcie (sekcja KONFIGURACJA).")
        return

    print(f"🔧 Rozpoczynam ręczne pobieranie {len(MANUAL_URLS)} meczów dla sezonu {SEASON}...")

    # Ustalamy ścieżki wyjściowe
    output_filename = f"{LEAGUE_PREFIX}{SEASON.replace('-', '')}.json"
    output_path = DATA_DIR / output_filename

    print(f"📂 Dane zostaną dopisane do pliku: {output_filename}")

    # Budowanie komendy
    command = [
        str(SCRAPER_PYTHON), "-m", "oddsharvester", "historic",
        "--season", SEASON,
        "-s", "football",
        "-l", LEAGUE,
        "-m", MARKETS,
        "--headless",
        "-o", str(output_path)
    ]

    # Dodajemy każdy URL z listy jako osobny argument --match-link
    for url in MANUAL_URLS:
        command.extend(["--match-link", url])

    # Uruchomienie
    print("🚀 Uruchamiam dociąganie danych...")
    
    # Nazwa pliku logu dla ręcznego uruchomienia
    manual_log_filename = f"manual_retry_log_{SEASON}.txt"
    manual_log_path = LOGS_DIR / manual_log_filename
    print(f"📝 Logi z tej operacji trafią do: {manual_log_filename}")

    with open(manual_log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command, 
            text=True,
            cwd=str(SCRAPER_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

    if result.returncode == 0:
        print(f"✅ Sukces! Dociągnięto mecze.")
    else:
        print(f"❌ Coś poszło nie tak. Sprawdź logi: {manual_log_filename}")

if __name__ == "__main__":
    run_manual_retry()