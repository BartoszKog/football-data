import subprocess
from pathlib import Path
from typing import List

# --- KONFIGURACJA ---
# 1. Wklej tutaj linki do nadchodzących meczów, które chcesz pobrać (w cudzysłowach, oddzielone przecinkami)
MATCH_LINKS: List[str] = [
    # "https://www.oddsportal.com/football/poland/ekstraklasa/rakow-czestochowa-lech-poznan-W4aM1b23/",
    "https://www.oddsportal.com/football/poland/division-1/chrobry-glogow-puszcza-M5HOknmp/",
    "https://www.oddsportal.com/football/poland/division-1/wisla-legnica-8t276gkL/",
    "https://www.oddsportal.com/football/poland/division-1/pogon-grodzisk-mazowiecki-polonia-warszawa-IimPgG5F/",
    "https://www.oddsportal.com/football/poland/division-1/ruch-chorzow-stal-mielec-p2X3nYck/",
    "https://www.oddsportal.com/football/poland/division-1/s-rzeszow-leczna-6HLCpCS1/",
    "https://www.oddsportal.com/football/poland/division-1/pogon-siedlce-pruszkow-EJtYifzS/",
    "https://www.oddsportal.com/football/poland/division-1/tychy-slask-wroclaw-UVKWmQHd/",
    "https://www.oddsportal.com/football/poland/division-1/lks-lodz-odra-opole-pvkHexy3/",
    "https://www.oddsportal.com/football/poland/division-1/ks-wieczysta-krakow-polonia-bytom-CE8b8FK8/"
]

# 2. Jeśli NIE podajesz linków, możesz pobrać mecze po dacie (lub kilku datach) + lidze
#    Format daty: YYYYMMDD (np. "20250320")
DATES: List[str] = [
    # "20250320",
]

# 3. Ustaw podstawowe parametry dla scrapera
SPORT = "football"
LEAGUE = "poland-1-liga"
MARKETS = "1x2,btts,over_under_2_5"

# 4. Prefiks nazwy pliku wyjściowego (np. dla ligi 1. ligi)
LEAGUE_PREFIX = "1ligaUpcoming_"

# --- USTAWIENIA ŚCIEŻEK ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Folder football-data/
DATA_DIR = BASE_DIR / "data" / "raw"
LOGS_DIR = BASE_DIR / "logs"

# Ścieżka do Pythona wewnątrz OddsHarvester
SCRAPER_DIR = BASE_DIR / "OddsHarvester"
SCRAPER_PYTHON = SCRAPER_DIR / ".venv" / "Scripts" / "python.exe"


def _run_command(command: list[str], log_filename: str) -> None:
    """Uruchamia podaną komendę w katalogu scrapera i zapisuje logi do pliku."""
    if not SCRAPER_PYTHON.exists():
        print(f"❌ BŁĄD: Nie znaleziono pythona w {SCRAPER_PYTHON}")
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOGS_DIR / log_filename
    print(f"📝 Logi z tej operacji trafią do: {log_filename}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            text=True,
            cwd=str(SCRAPER_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    if result.returncode == 0:
        print("✅ Sukces! Dane zostały pobrane.")
    else:
        print(f"❌ Coś poszło nie tak. Sprawdź logi: {log_filename}")


def run_upcoming_by_links() -> None:
    """Pobiera nadchodzące mecze na podstawie listy linków MATCH_LINKS."""
    if not MATCH_LINKS:
        print("⚠️ Lista MATCH_LINKS jest pusta. Uzupełnij linki w sekcji KONFIGURACJA.")
        return

    print(f"🔧 Rozpoczynam pobieranie {len(MATCH_LINKS)} nadchodzących meczów z linków...")

    output_filename = f"{LEAGUE_PREFIX}links.json"
    output_path = DATA_DIR / output_filename
    print(f"📂 Dane trafią do pliku: {output_filename}")

    command = [
        str(SCRAPER_PYTHON),
        "-m",
        "oddsharvester",
        "upcoming",
        "-s",
        SPORT,
        "-m",
        MARKETS,
        "--headless",
        "-o",
        str(output_path),
    ]

    # Dodajemy każdy URL z listy jako osobny argument --match-link
    for url in MATCH_LINKS:
        command.extend(["--match-link", url])

    log_filename = "upcoming_manual_links_log.txt"
    _run_command(command, log_filename)


def run_upcoming_by_dates() -> None:
    """Pobiera nadchodzące mecze na podstawie dat (YYYYMMDD) i ligi."""
    if not DATES:
        print("⚠️ Lista DATES jest pusta. Uzupełnij daty w sekcji KONFIGURACJA.")
        return

    for date_str in DATES:
        print(f"🔧 Pobieram nadchodzące mecze dla daty {date_str} (liga: {LEAGUE})...")

        output_filename = f"{LEAGUE_PREFIX}{date_str}.json"
        output_path = DATA_DIR / output_filename
        print(f"📂 Dane trafią do pliku: {output_filename}")

        command = [
            str(SCRAPER_PYTHON),
            "-m",
            "oddsharvester",
            "upcoming",
            "-s",
            SPORT,
            "-d",
            date_str,
            "-l",
            LEAGUE,
            "-m",
            MARKETS,
            "--headless",
            "-o",
            str(output_path),
        ]

        log_filename = f"upcoming_date_{date_str}_log.txt"
        _run_command(command, log_filename)


def main() -> None:
    """
    Główne wejście skryptu.

    Logika:
    - Jeśli podano MATCH_LINKS (niepusta lista) -> pobiera po linkach.
    - Jeśli MATCH_LINKS jest puste, ale DATES nie są puste -> pobiera po datach.
    - Jeśli obie listy są niepuste -> uruchamia obie metody (najpierw linki, potem daty).
    """
    has_links = bool(MATCH_LINKS)
    has_dates = bool(DATES)

    if not has_links and not has_dates:
        print("⚠️ Ani MATCH_LINKS, ani DATES nie są ustawione. Uzupełnij konfigurację w skrypcie.")
        return

    if has_links:
        run_upcoming_by_links()

    if has_dates:
        run_upcoming_by_dates()


if __name__ == "__main__":
    main()

