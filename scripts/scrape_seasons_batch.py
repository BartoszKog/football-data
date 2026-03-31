import subprocess
import time
from pathlib import Path
import sys
import os

# --- KONFIGURACJA ---
SEASONS_TO_SCRAPE = [
    # "2023-2024",
    # "2022-2023",
    # "2021-2022",
    # "2020-2021",
    "current"
]
LEAGUE = "poland-1-liga"
MARKETS = "1x2,btts,over_under_2_5"

# --- USTAWIENIA ŚCIEŻEK ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Folder football-data/
DATA_DIR = BASE_DIR / "data" / "raw"
LOGS_DIR = BASE_DIR / "logs"

# Ścieżka do Pythona wewnątrz OddsHarvester
# Zakładamy, że jesteś na Windows (folder Scripts)
SCRAPER_DIR = BASE_DIR / "OddsHarvester"
SCRAPER_PYTHON = SCRAPER_DIR / ".venv" / "Scripts" / "python.exe"

# prefix nazwy pliku na nazwę ligii
LEAGUE_PREFIX = "1liga300326current_"

# --------------------

def run_batch_scraping():
    # Sprawdzenie czy scraper ma swoje środowisko
    if not SCRAPER_PYTHON.exists():
        print(f"❌ BŁĄD: Nie znaleziono pythona w {SCRAPER_PYTHON}")
        print("Wejdź do folderu OddsHarvester i wpisz: uv sync")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Uruchamiam zewnętrzny scraper z: {SCRAPER_PYTHON}")

    for season in SEASONS_TO_SCRAPE:
        print(f"\n--- ⏳ Pobieranie sezonu {season} ---")
        
        output_filename = f"{LEAGUE_PREFIX}{season.replace('-', '')}.json"
        output_path = DATA_DIR / output_filename
        log_path = LOGS_DIR / f"scrape_log_{season}.txt"

        # Komenda uruchamiajaca moduł w tamtym środowisku
        command = [
            str(SCRAPER_PYTHON), "-m", "oddsharvester", "historic",
            "-s", "football",
            "-l", LEAGUE,
            "--season", season,
            "-m", MARKETS,
            "--headless",
            "-o", str(output_path)
        ]

        with open(log_path, "w", encoding="utf-8") as log_file:
            # cwd=SCRAPER_DIR sprawia, że skrypt "myśli", że jest w tamtym folderze
            result = subprocess.run(
                command, 
                stdout=log_file, 
                stderr=subprocess.STDOUT, 
                text=True,
                cwd=str(SCRAPER_DIR) 
            )

        if result.returncode == 0:
            print(f"✅ Sezon {season} gotowy!")
        else:
            print(f"❌ Błąd w sezonie {season}. Zobacz logi: {log_path}")

        time.sleep(5)

if __name__ == "__main__":
    run_batch_scraping()