import subprocess
import time
from pathlib import Path
import re
import ast
import sys

# --- KONFIGURACJA ---
# Wpisz sezony, dla których chcesz naprawić błędy (musi istnieć plik logów w folderze logs/)
SEASONS_TO_RETRY = [
    "2023-2024",
    "2022-2023",
    "2021-2022",
    "2020-2021",
]
LEAGUE = "poland-1-liga"
MARKETS = "1x2,btts,over_under_2_5"

# --- USTAWIENIA ŚCIEŻEK ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Folder football-data/
DATA_DIR = BASE_DIR / "data" / "raw"
LOGS_DIR = BASE_DIR / "logs"

# Ścieżka do Pythona wewnątrz OddsHarvester (tak samo jak w poprzednim skrypcie)
SCRAPER_DIR = BASE_DIR / "OddsHarvester"
SCRAPER_PYTHON = SCRAPER_DIR / ".venv" / "Scripts" / "python.exe"

# --------------------

def get_failed_urls(log_path: Path) -> list[str]:
    """Parsuje plik logów i szuka listy Failed URLs."""
    if not log_path.exists():
        print(f"⚠️  Brak pliku logów: {log_path}")
        return []


    with open(log_path, "r", encoding="cp1250") as f:
        content = f.read()
        
        # Szukamy linii: Failed URLs: ['http...', 'http...']
        # Używamy regexa, żeby znaleźć listę na końcu pliku
        match = re.search(r"Failed URLs:\s*(\[.*?\])", content, re.DOTALL)
        
        if match:
            list_str = match.group(1)
            # Bezpieczna konwersja tekstu "['url1', 'url2']" na prawdziwą listę Pythona
            urls = ast.literal_eval(list_str)
            return urls
        else:
            return []

def run_retry_script():
    # Sprawdzenie środowiska
    if not SCRAPER_PYTHON.exists():
        print(f"❌ BŁĄD: Nie znaleziono pythona w {SCRAPER_PYTHON}")
        return

    print(f"🔧 Rozpoczynam naprawianie brakujących meczów...")

    for season in SEASONS_TO_RETRY:
        print(f"\n--- Analiza sezonu {season} ---")
        
        # Ustalamy ścieżki
        log_filename = f"scrape_log_{season}.txt"
        log_path = LOGS_DIR / log_filename
        
        output_filename = f"1liga_{season.replace('-', '')}.json"
        output_path = DATA_DIR / output_filename

        # 1. Pobierz brakujące linki z logów
        failed_urls = get_failed_urls(log_path)

        if not failed_urls:
            print(f"✅ Brak błędów do naprawienia w sezonie {season} (lub nie znaleziono logu).")
            continue

        print(f"🔍 Znaleziono {len(failed_urls)} brakujących meczów.")
        print(f"📂 Dane zostaną dopisane do: {output_filename}")

        # 2. Budowanie komendy z wieloma flagami --match-link
        # Struktura: python -m oddsharvester historic --match-link URL1 --match-link URL2 ...
        command = [
            str(SCRAPER_PYTHON), "-m", "oddsharvester", "historic",
            "--season", season,
            "-s", "football",
            "-l", LEAGUE,
            "-m", MARKETS,
            "--headless",
            "-o", str(output_path)
        ]

        # Dodajemy każdy URL jako osobny argument --match-link
        for url in failed_urls:
            command.extend(["--match-link", url])

        # 3. Uruchomienie
        print("🚀 Uruchamiam dociąganie danych...")
        
        retry_log_filename = f"retry_log_{season}.txt"
        retry_log_path = LOGS_DIR / retry_log_filename
        print(f"📝 Logi z tej operacji trafią do: {retry_log_filename}")

        with open(retry_log_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command, 
                text=True,
                cwd=str(SCRAPER_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT
            )

        if result.returncode == 0:
            print(f"✅ Sukces! Dociągnięto mecze dla sezonu {season}.")
        else:
            print(f"❌ Coś poszło nie tak przy naprawianiu {season}.")


        time.sleep(2)

    print("\n🎉 Zakończono procedurę naprawczą!")

if __name__ == "__main__":
    run_retry_script()