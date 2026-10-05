"""Telegram front-end for the Asana task tracker. Runs on Windows, Linux, macOS or Android (Termux)."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

import asana_client as ac

log = logging.getLogger("tracker")
HELP = (
    "Asana task tracker\n"
    "/tasks - all my open tasks\n"
    "/today - due today\n"
    "/overdue - past due\n"
    "/find <text> - search my tasks\n"
    "/add <title> [| YYYY-MM-DD] - create a task\n"
    "Tap a task's button to mark it done. A digest is sent daily."
)


def fmt(t: dict) -> str:
    due = f" (due {t['due_on']})" if t.get("due_on") else ""
    return f"• {t['name']}{due}"


def keyboard(tasks: list[dict]) -> InlineKeyboardMarkup | None:
    rows = [[InlineKeyboardButton(f"✅ {t['name'][:40]}", callback_data=f"done:{t['gid']}")] for t in tasks[:20]]
    return InlineKeyboardMarkup(rows) if rows else None


def allowed(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    return bool(user and user.id in ctx.bot_data["allowed"])


async def reply_tasks(update: Update, ctx: ContextTypes.DEFAULT_TYPE, title: str, tasks: list[dict]) -> None:
    if not tasks:
        await update.message.reply_text(f"{title}: nothing 🎉")
        return
    text = f"{title} ({len(tasks)}):\n" + "\n".join(fmt(t) for t in tasks[:20])
    if len(tasks) > 20:
        text += f"\n…and {len(tasks) - 20} more"
    await update.message.reply_text(text, reply_markup=keyboard(tasks))


def guarded(fn):
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not allowed(update, ctx):
            if update.effective_message:
                await update.effective_message.reply_text("Not authorised.")
            return
        try:
            await fn(update, ctx)
        except ac.AsanaError as e:
            await update.effective_message.reply_text(f"⚠️ {e}")
        except Exception:
            log.exception("handler failed")
            await update.effective_message.reply_text("⚠️ Something went wrong.")
    return wrapper


def client(ctx) -> ac.AsanaClient:
    return ctx.bot_data["asana"]


@guarded
async def start(update, ctx):
    await update.message.reply_text(HELP)


@guarded
async def tasks_cmd(update, ctx):
    await reply_tasks(update, ctx, "Open tasks", await asyncio.to_thread(client(ctx).my_tasks))


@guarded
async def today_cmd(update, ctx):
    today = dt.datetime.now(ctx.bot_data["tz"]).date()
    tasks = ac.due_today(await asyncio.to_thread(client(ctx).my_tasks), today)
    await reply_tasks(update, ctx, "Due today", tasks)


@guarded
async def overdue_cmd(update, ctx):
    today = dt.datetime.now(ctx.bot_data["tz"]).date()
    tasks = ac.overdue(await asyncio.to_thread(client(ctx).my_tasks), today)
    await reply_tasks(update, ctx, "Overdue", tasks)


@guarded
async def find_cmd(update, ctx):
    text = " ".join(ctx.args).strip()
    if not text:
        await update.message.reply_text("Usage: /find <text>")
        return
    await reply_tasks(update, ctx, f"Matches for '{text}'", ac.find(await asyncio.to_thread(client(ctx).my_tasks), text))


@guarded
async def add_cmd(update, ctx):
    raw = " ".join(ctx.args)
    name, _, due = (p.strip() for p in raw.partition("|"))
    if not name:
        await update.message.reply_text("Usage: /add <title> [| YYYY-MM-DD]")
        return
    if due:
        try:
            dt.date.fromisoformat(due)
        except ValueError:
            await update.message.reply_text("Date must look like 2026-12-31")
            return
    t = await asyncio.to_thread(client(ctx).create_task, name, due or None)
    await update.message.reply_text(f"Created:\n{fmt(t)}\n{t.get('permalink_url', '')}")


async def done_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not allowed(update, ctx):
        await q.answer("Not authorised", show_alert=True)
        return
    gid = q.data.split(":", 1)[1]
    try:
        t = await asyncio.to_thread(client(ctx).complete_task, gid)
    except ac.AsanaError as e:
        await q.answer(f"⚠️ {e}", show_alert=True)
        return
    await q.answer(f"Done: {t['name'][:40]}")
    # drop the completed task's button
    kb = [r for r in q.message.reply_markup.inline_keyboard if r[0].callback_data != q.data]
    await q.edit_message_reply_markup(InlineKeyboardMarkup(kb) if kb else None)


async def digest(ctx: ContextTypes.DEFAULT_TYPE):
    today = dt.datetime.now(ctx.bot_data["tz"]).date()
    tasks = await asyncio.to_thread(client(ctx).my_tasks)
    due, late = ac.due_today(tasks, today), ac.overdue(tasks, today)
    if not due and not late:
        return
    parts = []
    if late:
        parts.append(f"Overdue ({len(late)}):\n" + "\n".join(fmt(t) for t in late[:15]))
    if due:
        parts.append(f"Due today ({len(due)}):\n" + "\n".join(fmt(t) for t in due[:15]))
    for uid in ctx.bot_data["allowed"]:
        await ctx.bot.send_message(uid, "☀️ Daily digest\n\n" + "\n\n".join(parts), reply_markup=keyboard(late + due))


def build_app() -> Application:
    load_dotenv()
    env = os.environ
    for key in ("ASANA_TOKEN", "TELEGRAM_BOT_TOKEN", "ALLOWED_TELEGRAM_USER_IDS"):
        if not env.get(key):
            raise SystemExit(f"Missing {key} - copy .env.example to .env and fill it in.")
    app = Application.builder().token(env["TELEGRAM_BOT_TOKEN"]).build()
    tz = ZoneInfo(env.get("TZ_NAME", "UTC"))
    app.bot_data.update(
        asana=ac.AsanaClient(env["ASANA_TOKEN"], env.get("ASANA_WORKSPACE_GID")),
        allowed={int(x) for x in env["ALLOWED_TELEGRAM_USER_IDS"].split(",") if x.strip()},
        tz=tz,
    )
    for name, fn in [("start", start), ("help", start), ("tasks", tasks_cmd), ("today", today_cmd),
                     ("overdue", overdue_cmd), ("find", find_cmd), ("add", add_cmd)]:
        app.add_handler(CommandHandler(name, fn))
    app.add_handler(CallbackQueryHandler(done_button, pattern=r"^done:\d+$"))
    hh, mm = (int(x) for x in env.get("DIGEST_TIME", "08:30").split(":"))
    app.job_queue.run_daily(digest, dt.time(hh, mm, tzinfo=tz))
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_app().run_polling()
