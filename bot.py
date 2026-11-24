import os
import sys
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque
from typing import Dict, List

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    server.serve_forever()

threading.Thread(target=run_server, daemon=True).start()

# --- Poniżej zostawiasz resztę swojego kodu (import discord, inicjalizację bota itp.) ---
import os
import sys
import logging
from collections import deque
from typing import Dict, List

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DiscordAIBot")

# Wczytanie zmiennych środowiskowych z pliku .env
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Główny model oraz lista modeli zapasowych w przypadku przeciążenia (503 UNAVAILABLE / 429)
PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
BACKUP_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

MAX_HISTORY_PER_CHANNEL = 10
DISCORD_MAX_MSG_LEN = 1950  # Bezpieczny margines poniżej limitu 2000 znaków Discorda

# Inicjalizacja klienta Google GenAI SDK
# Używamy api_version='v1beta' oraz HttpRetryOptions z krótkim czasem ponowień,
# aby w razie błędu 503 bot błyskawicznie przełączał się na model rezerwowy.
gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "twoj_gemini_api_key_tutaj":
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=0.5)
        )
    )


class ChannelHistoryManager:
    """Zarządza pamięcią ostatnich wiadomości dla poszczególnych kanałów (kontekst rozmowy)."""

    def __init__(self, max_messages: int = MAX_HISTORY_PER_CHANNEL):
        self.max_messages = max_messages
        self._history: Dict[int, deque] = {}

    def get_history(self, channel_id: int) -> deque:
        if channel_id not in self._history:
            self._history[channel_id] = deque(maxlen=self.max_messages)
        return self._history[channel_id]

    def add_user_message(self, channel_id: int, author: str, text: str):
        history = self.get_history(channel_id)
        history.append({
            "role": "user",
            "author": author,
            "text": f"{author}: {text}"
        })

    def add_model_message(self, channel_id: int, text: str):
        history = self.get_history(channel_id)
        history.append({
            "role": "model",
            "author": "Gemini",
            "text": text
        })

    def clear(self, channel_id: int):
        if channel_id in self._history:
            self._history[channel_id].clear()


history_manager = ChannelHistoryManager(max_messages=MAX_HISTORY_PER_CHANNEL)


def split_message(text: str, max_length: int = DISCORD_MAX_MSG_LEN) -> List[str]:
    """
    Dzieli długi tekst na części nieprzekraczające zadanego limitu (domyślnie 1950 znaków).
    Zachowuje układ linii tam gdzie to możliwe.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = []
    current_len = 0

    for line in text.splitlines(keepends=True):
        if len(line) > max_length:
            for i in range(0, len(line), max_length):
                part = line[i : i + max_length]
                if current_len + len(part) > max_length:
                    if current_chunk:
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                current_chunk.append(part)
                current_len += len(part)
        else:
            if current_len + len(line) > max_length:
                chunks.append("".join(current_chunk))
                current_chunk = [line]
                current_len = len(line)
            else:
                current_chunk.append(line)
                current_len += len(line)

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


async def generate_gemini_response(channel_id: int, author: str, prompt: str) -> str:
    """
    Generuje odpowiedź AI z uwzględnieniem pamięci kontekstu kanału (ostatnie 10 wiadomości).
    Zawiera automatyczną obsługę błędu 503 UNAVAILABLE / 429 z przełączaniem na modele zapasowe.
    """
    global gemini_client

    if not gemini_client:
        current_key = os.getenv("GEMINI_API_KEY")
        if current_key and current_key != "twoj_gemini_api_key_tutaj":
            gemini_client = genai.Client(
                api_key=current_key,
                http_options=types.HttpOptions(
                    api_version="v1beta",
                    retry_options=types.HttpRetryOptions(attempts=2, initial_delay=0.5)
                )
            )
        else:
            return "⚠️ Błąd: Brak poprawnego klucza `GEMINI_API_KEY` w pliku `.env`. Uzupełnij go, aby korzystać z Gemini!"

    # 1. Dodanie zapytania użytkownika do historii kanału
    history_manager.add_user_message(channel_id=channel_id, author=author, text=prompt)

    # 2. Przygotowanie kontekstu dla SDK google-genai
    raw_history = list(history_manager.get_history(channel_id))
    contents: List[types.Content] = []

    for msg in raw_history:
        role = msg["role"]
        text = msg["text"]
        # Wymóg naprzemienności ról w Gemini: łączymy wiadomości o tej samej roli
        if contents and contents[-1].role == role:
            contents[-1].parts[0].text += f"\n\n{text}"
        else:
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    # Gemini API wymaga, aby konwersacja zaczynała się od wiadomości o roli 'user'
    while contents and contents[0].role != "user":
        contents.pop(0)

    if not contents:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    # Zbudowanie kolejki modeli: najpierw model główny, potem modele zapasowe
    models_to_try = [PRIMARY_MODEL]
    for model_name in BACKUP_MODELS:
        if model_name not in models_to_try:
            models_to_try.append(model_name)

    last_error = None
    for idx, model_candidate in enumerate(models_to_try):
        try:
            logger.info(f"Generowanie odpowiedzi: model '{model_candidate}' (próba {idx+1}/{len(models_to_try)})")
            response = await gemini_client.aio.models.generate_content(
                model=model_candidate,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "Jesteś pomocnym, inteligentnym i przyjaznym asystentem AI na serwerze Discord. "
                        "Działasz w oparciu o model Gemini od Google. "
                        "Odpowiadaj wyczerpująco, logicznie i czytelnie. Używaj formatowania Markdown "
                        "(nagłówki, listy wypunktowane, pogrubienia, bloki kodu z oznaczeniem języka). "
                        "Dostosuj język do rozmówcy (domyślnie język polski)."
                    ),
                    temperature=0.7,
                ),
            )

            answer = response.text
            if not answer:
                answer = "Przepraszam, nie udało mi się wygenerować odpowiedzi (możliwa blokada przez filtry bezpieczeństwa)."

            # 3. Zapisanie odpowiedzi modelu w historii kanału
            history_manager.add_model_message(channel_id=channel_id, text=answer)
            if idx > 0:
                logger.info(f"Sukces! Odpowiedź wygenerowano pomyślnie za pomocą modelu zapasowego: '{model_candidate}'")
            return answer

        except Exception as e:
            last_error = e
            err_str = str(e)
            is_overloaded = ("503" in err_str) or ("UNAVAILABLE" in err_str) or ("429" in err_str) or ("RESOURCE_EXHAUSTED" in err_str)
            logger.warning(f"Błąd przy modelu '{model_candidate}': {e}")

            # Jeśli model jest przeciążony (503 lub 429), automatycznie przełączamy na następny model zapasowy
            if is_overloaded and idx < len(models_to_try) - 1:
                next_model = models_to_try[idx + 1]
                logger.warning(
                    f"⚠️ Model '{model_candidate}' jest chwilowo przeciążony (błąd 503/429). "
                    f"Automatycznie przełączam na zapasowy model: '{next_model}'..."
                )
                continue
            elif idx < len(models_to_try) - 1:
                # Inny błąd (np. 404 lub tymczasowy) - również podejmujemy próbę z modelem zapasowym
                logger.warning(f"Próbuję model zapasowy z powodu błędu: {e}")
                continue
            else:
                break

    logger.error(f"Wszystkie próby modeli zakończyły się błędem: {last_error}", exc_info=True)
    return f"⚠️ Wystąpił błąd podczas generowania odpowiedzi AI (wszystkie modele przeciążone): `{last_error}`"


# Konfiguracja uprawnień (Intents)
intents = discord.Intents.default()
intents.message_content = True  # Wymagane do odczytu treści wiadomości przy bezpośrednim oznaczeniu @bot


class GeminiDiscordBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        # Globalna synchronizacja komend slash
        logger.info("Synchronizacja globalnych komend slash...")
        try:
            synced = await self.tree.sync()
            logger.info(f"Pomyślnie zsynchronizowano globalnie {len(synced)} komend(y) slash.")
        except Exception as e:
            logger.error(f"Błąd podczas globalnej synchronizacji komend slash: {e}")

    async def on_ready(self):
        logger.info(f"Zalogowano pomyślnie jako: {self.user} (ID: {self.user.id})")
        logger.info(f"Główny model AI: {PRIMARY_MODEL}")
        logger.info(f"Modele zapasowe (fallback 503): {', '.join(BACKUP_MODELS)}")

        # Błyskawiczna synchronizacja komend slash dla każdego połączonego serwera
        logger.info(f"Wykryto {len(self.guilds)} serwer(ów). Rejestruję komendy slash dla serwerów...")
        for guild in self.guilds:
            try:
                self.tree.copy_global_to(guild=guild)
                synced_guild = await self.tree.sync(guild=guild)
                logger.info(f"✅ Błyskawicznie zarejestrowano {len(synced_guild)} komend slash na serwerze: '{guild.name}' (ID: {guild.id})")
            except Exception as e:
                logger.warning(f"Nie udało się zsynchronizować komend dla serwera '{guild.name}': {e}")

        logger.info("Bot jest gotowy do odbierania pytań!")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="/zapytaj lub @mnie"
            )
        )

    async def on_guild_join(self, guild: discord.Guild):
        logger.info(f"Bot dołączył do nowego serwera: '{guild.name}' (ID: {guild.id}). Rejestruję komendy...")
        try:
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info(f"✅ Błyskawicznie zarejestrowano {len(synced)} komend slash na serwerze: '{guild.name}'")
        except Exception as e:
            logger.warning(f"Błąd synchronizacji po dołączeniu do serwera '{guild.name}': {e}")


bot = GeminiDiscordBot()


# 1. Komenda Slash: /zapytaj [pytanie]
@bot.tree.command(name="zapytaj", description="Zadaj pytanie asystentowi AI (Gemini)")
@app_commands.describe(pytanie="Treść Twojego pytania do asystenta")
async def zapytaj(interaction: discord.Interaction, pytanie: str):
    logger.info(f"[Slash /zapytaj] Użytkownik {interaction.user} w #{interaction.channel}: {pytanie[:60]}...")

    # Informujemy Discord, że odpowiedź jest przygotowywana (zapobiega timeoutowi 3s)
    await interaction.response.defer(thinking=True)

    # Próba dodania emotki ⏳ podczas generowania odpowiedzi
    orig_msg = None
    try:
        orig_msg = await interaction.original_response()
        await orig_msg.add_reaction("⏳")
    except Exception as e:
        logger.debug(f"Info o reakcji na interakcję: {e}")

    try:
        response_text = await generate_gemini_response(
            channel_id=interaction.channel_id,
            author=interaction.user.display_name,
            prompt=pytanie,
        )
    finally:
        if orig_msg:
            try:
                await orig_msg.remove_reaction("⏳", bot.user)
            except Exception:
                pass

    # Podział odpowiedzi na części, jeśli przekracza 2000 znaków
    chunks = split_message(response_text)
    await interaction.followup.send(chunks[0])
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


# Komenda Slash: /wyczysc (pomocnicza do zresetowania pamięci danego kanału)
@bot.tree.command(name="wyczysc", description="Wyczyść pamięć ostatnich wiadomości w tym kanale")
async def wyczysc(interaction: discord.Interaction):
    history_manager.clear(interaction.channel_id)
    await interaction.response.send_message(
        "🧹 Pamięć ostatnich wiadomości w tym kanale została wyczyszczona!",
        ephemeral=True
    )


# Komenda prefiksowa !sync do ręcznego wymuszenia synchronizacji z poziomu czatu
@bot.command(name="sync")
@commands.has_permissions(administrator=True)
async def sync_command(ctx: commands.Context):
    """Ręczna synchronizacja komend slash dla bieżącego serwera."""
    status_msg = await ctx.send("⏳ Trwa natychmiastowa synchronizacja komend slash dla tego serwera...")
    try:
        bot.tree.copy_global_to(guild=ctx.guild)
        synced = await bot.tree.sync(guild=ctx.guild)
        await status_msg.edit(
            content=f"✅ Zsynchronizowano pomyślnie {len(synced)} komend slash dla tego serwera!\n"
                    f"*(Jeśli komendy nie pojawią się od razu w menu, odśwież aplikację Discord skrótem `Ctrl+R`)*"
        )
    except Exception as e:
        await status_msg.edit(content=f"❌ Błąd synchronizacji komend: `{e}`")


# 2. Bezpośrednie oznaczenie na czacie: @nazwa_bota
@bot.event
async def on_message(message: discord.Message):
    # Ignorujemy wiadomości od botów (w tym własne)
    if message.author.bot:
        return

    # Sprawdzamy czy bot został bezpośrednio oznaczony (z pominięciem @everyone i @here)
    if bot.user in message.mentions and not message.mention_everyone:
        logger.info(f"[Mention] Użytkownik {message.author} w #{message.channel}: {message.content[:60]}...")

        # Usunięcie oznaczenia bota z treści wiadomości
        clean_prompt = message.content
        for mention in [f"<@{bot.user.id}>", f"<@!{bot.user.id}>"]:
            clean_prompt = clean_prompt.replace(mention, "")
        clean_prompt = clean_prompt.strip()

        if not clean_prompt:
            await message.reply(
                "Cześć! W czym mogę Ci dzisiaj pomóc? Zadaj mi pytanie oznaczając mnie na czacie lub użyj komendy `/zapytaj`.",
                mention_author=False
            )
            return

        # Reakcja emotką ⏳ na czas generowania odpowiedzi
        try:
            await message.add_reaction("⏳")
        except Exception as e:
            logger.warning(f"Nie udało się dodać reakcji ⏳: {e}")

        try:
            async with message.channel.typing():
                response_text = await generate_gemini_response(
                    channel_id=message.channel.id,
                    author=message.author.display_name,
                    prompt=clean_prompt
                )
        finally:
            try:
                await message.remove_reaction("⏳", bot.user)
            except Exception as e:
                logger.warning(f"Nie udało się usunąć reakcji ⏳: {e}")

        # Podział odpowiedzi na części, jeśli przekracza 2000 znaków
        chunks = split_message(response_text)
        await message.reply(chunks[0], mention_author=False)
        for chunk in chunks[1:]:
            await message.channel.send(chunk)

    # Przetwarzanie standardowych prefiksów komend (np. !sync)
    await bot.process_commands(message)


def main():
    if not DISCORD_TOKEN or DISCORD_TOKEN == "twoj_discord_token_tutaj":
        print("\n" + "="*70)
        print("❌ BŁĄD: Nie znaleziono poprawnego DISCORD_TOKEN w pliku .env!")
        print("1. Otwórz plik .env w edytorze.")
        print("2. Wklej swój token w miejscu: DISCORD_TOKEN=...")
        print("3. Upewnij się, że uzupełniłeś także GEMINI_API_KEY.")
        print("4. Szczegóły znajdziesz w pliku README.md")
        print("="*70 + "\n")
        sys.exit(1)

    try:
        bot.run(DISCORD_TOKEN)
    except discord.errors.LoginFailure:
        print("\n❌ Błąd logowania: Podany DISCORD_TOKEN jest nieprawidłowy!")
        sys.exit(1)
    except discord.errors.PrivilegedIntentsRequired:
        print("\n❌ BŁĄD UPRAWNIEŃ (Privileged Intents):")
        print("Musisz włączyć 'MESSAGE CONTENT INTENT' w Discord Developer Portal:")
        print("1. Wejdź na https://discord.com/developers/applications")
        print("2. Wybierz swoją aplikację -> zakładka 'Bot'")
        print("3. W sekcji 'Privileged Gateway Intents' włącz opcję 'MESSAGE CONTENT INTENT'")
        print("4. Kliknij 'Save Changes' i uruchom bota ponownie.\n")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Nieoczekiwany błąd: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
