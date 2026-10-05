# Asana Task Tracker Bot

A Telegram bot that tracks your Asana tasks. Use it from the Telegram app on **Android** and **Windows**.
The bot process runs on a Windows PC (or Android via Termux).

## Commands
| Command | Does |
|---|---|
| `/tasks` | Open tasks assigned to you (tap a button to complete one) |
| `/today`, `/overdue` | Filtered views |
| `/find <text>` | Search your open tasks |
| `/add <title> [\| YYYY-MM-DD]` | Create a task assigned to you |
| daily digest | Overdue and due-today tasks, pushed at `DIGEST_TIME` |

## Setup
1. **Asana token**: Asana > profile photo > Settings > Apps > Developer apps > *New access token*.
2. **Telegram bot**: message [@BotFather](https://t.me/BotFather), `/newbot`, copy the token.
3. **Your Telegram ID**: message [@userinfobot](https://t.me/userinfobot). Only IDs in `ALLOWED_TELEGRAM_USER_IDS` can use the bot.
4. Copy `.env.example` to `.env` and fill it in.

### Windows
Double-click `run_windows.bat` (creates a venv, installs deps, starts the bot). Needs Python 3.10+.
To run at login, add a shortcut to `run_windows.bat` in `shell:startup`.

### Android (host the bot on the phone, optional)
Install Termux (from F-Droid), then:
```
pkg install python git && pip install -r requirements.txt && python bot.py
```
Or just keep it on the Windows PC/any server and use the Telegram app on the phone.

## Tests
`python -m pytest`

## Notes
- The bot uses long polling, so there's no public URL or port forwarding needed.
- Your PAT acts as you, so keep `.env` private (don't commit it).
