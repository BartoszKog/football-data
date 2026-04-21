import subprocess
import time
from pathlib import Path
import sys

# --- KONFIGURACJA ---
# 1. Wpisz sezon, do którego mają trafić mecze
SEASON = "current"

# 2. Wklej tutaj linki do meczów, które chcesz pobrać ręcznie (w cudzysłowach, oddzielone przecinkami)
MANUAL_URLS = [
    "https://www.oddsportal.com/pl/football/h2h/pogon-siedlce-d6IVB42i/slask-wroclaw-E1Oxemse/",
    "https://www.oddsportal.com/pl/football/h2h/ks-wieczysta-krakow-ImhQf1z6/lks-lodz-ShUKWHDG/",
    "https://www.oddsportal.com/football/h2h/pruszkow-K8YxVlJK/tychy-baXtU8YQ/",
    "https://www.oddsportal.com/pl/football/h2h/gornik-leczna-ImAVQStr/wisla-krakow-rob20Q2Q/",
    "https://www.oddsportal.com/pl/football/h2h/odra-opole-C8Kg1k4R/stal-rzeszow-IkpY1q92/",
    "https://www.oddsportal.com/pl/football/h2h/polonia-warszawa-ADZjR3us/puszcza-dtqx13O8/",
    "https://www.oddsportal.com/pl/football/h2h/chrobry-glogow-vo6umLPj/ruch-chorzow-S4zygtuJ/",
    "https://www.oddsportal.com/pl/football/h2h/pogon-grodzisk-mazowiecki-4nBQsDq2/stal-mielec-pxXiDZkQ/",
    "https://www.oddsportal.com/pl/football/h2h/miedz-legnica-lnY5n2Xe/polonia-bytom-vRn3UqXP/",
    "https://www.oddsportal.com/pl/football/h2h/polonia-warszawa-ADZjR3us/slask-wroclaw-E1Oxemse/",
    "https://www.oddsportal.com/pl/football/h2h/odra-opole-C8Kg1k4R/puszcza-dtqx13O8/",
    "https://www.oddsportal.com/football/h2h/s-rzeszow-IkpY1q92/stal-mielec-pxXiDZkQ/",
    "https://www.oddsportal.com/pl/football/h2h/ks-wieczysta-krakow-ImhQf1z6/ruch-chorzow-S4zygtuJ/",
    "https://www.oddsportal.com/pl/football/h2h/chrobry-glogow-vo6umLPj/gornik-leczna-ImAVQStr/",
    "https://www.oddsportal.com/football/h2h/pogon-siedlce-d6IVB42i/tychy-baXtU8YQ/",
    "https://www.oddsportal.com/pl/football/h2h/polonia-bytom-vRn3UqXP/wisla-krakow-rob20Q2Q/",
    "https://www.oddsportal.com/pl/football/h2h/miedz-legnica-lnY5n2Xe/znicz-pruszkow-K8YxVlJK/",
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