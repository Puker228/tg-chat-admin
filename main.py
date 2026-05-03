import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urljoin

import httpx
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    bot_token: str
    ai_base_url: HttpUrl = HttpUrl("http://localhost:11434")
    ai_model: str = "gemma3:270m"
    backend_url: HttpUrl = HttpUrl("http://localhost:8000")


settings = Settings()

Intent = Literal["hello", "bye", "other"]
BACKEND_INTENTS: set[Intent] = {"hello", "bye"}


@dataclass(frozen=True)
class MessageContext:
    intent: Intent
    analysis: str


SUPPORT_REPLIES = [
    "Ваш запрос принят! Мы уже работаем над этим 💙",
    "Спасибо, что написали нам! Всё будет решено в ближайшее время 🌟",
    "Мы получили ваше сообщение и обязательно поможем! Держитесь 🤝",
    "Ваш вопрос важен для нас! Команда уже в курсе 🚀",
    "Спасибо за обращение! Мы на связи и не оставим вас без ответа 🙌",
    "Всё фиксируем, всё решим! Вы в надёжных руках 💪",
    "Ваша заявка уже у нас! Скоро всё будет хорошо ☀️",
    "Мы здесь, мы слышим вас, мы поможем! 🫂",
]

dp = Dispatcher()


def parse_message_context(raw_response: str, fallback_text: str) -> MessageContext:
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start == -1 or end == -1 or start > end:
        return MessageContext(intent="other", analysis=fallback_text)

    try:
        data = json.loads(raw_response[start : end + 1])
    except json.JSONDecodeError:
        return MessageContext(intent="other", analysis=fallback_text)

    intent = str(data.get("intent", "other")).strip().lower()
    if intent not in BACKEND_INTENTS:
        intent = "other"

    analysis = str(data.get("analysis", "")).strip() or fallback_text
    return MessageContext(intent=intent, analysis=analysis)


async def analyze_with_ollama(text: str) -> MessageContext:
    prompt = (
        "Проанализируй сообщение пользователя и определи его намерение.\n"
        "Если пользователь хочет поздороваться, передать привет или попросить кого-то передать привет, "
        'intent должен быть "hello".\n'
        "Если пользователь хочет попрощаться, передать прощание или попросить кого-то попрощаться, "
        'intent должен быть "bye".\n'
        'Во всех остальных случаях intent должен быть "other".\n'
        "Верни только валидный JSON без Markdown и пояснений в формате:\n"
        '{"intent":"hello|bye|other","analysis":"краткая суть сообщения на русском"}\n\n'
        f"Сообщение: {text}"
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.ai_base_url}api/generate",
            json={"model": settings.ai_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return parse_message_context(response.json()["response"].strip(), text)


def get_backend_url(intent: Intent) -> str | None:
    if intent not in BACKEND_INTENTS:
        return None

    return urljoin(str(settings.backend_url), f"/{intent}")


async def send_to_backend(user_message: str, context: MessageContext) -> bool:
    backend_url = get_backend_url(context.intent)
    if backend_url is None:
        logging.info("Backend notification skipped for intent: %s", context.intent)
        return False

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            backend_url,
            json={
                "message": user_message,
                "analysis": context.analysis,
                "intent": context.intent,
            },
        )
        response.raise_for_status()
        return True


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(
        f"Привет, {html.bold(message.from_user.full_name)}! Чем могу помочь?"
    )


@dp.message()
async def message_handler(message: Message) -> None:
    if not message.text:
        await message.answer("Пожалуйста, отправьте текстовое сообщение.")
        return

    try:
        context = await analyze_with_ollama(message.text)
        logging.info(
            "Ollama context: intent=%s analysis=%s",
            context.intent,
            context.analysis,
        )
    except Exception:
        logging.exception("Ollama request failed")
        context = MessageContext(intent="other", analysis=message.text)

    try:
        if await send_to_backend(message.text, context):
            logging.info("Backend notified successfully")
    except Exception:
        logging.exception("Backend request failed")

    await message.answer(random.choice(SUPPORT_REPLIES))


async def main() -> None:
    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
