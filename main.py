import asyncio
import logging
import random
import sys

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
    backend_url: HttpUrl = HttpUrl("http://localhost:8000/hello")


settings = Settings()

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


async def analyze_with_ollama(text: str) -> str:
    prompt = (
        "Проанализируй сообщение пользователя и в одном коротком предложении опиши суть запроса. "
        "Отвечай только на русском языке, без лишних слов.\n\n"
        f"Сообщение: {text}\n\nСуть запроса:"
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.ai_base_url}api/generate",
            json={"model": settings.ai_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"].strip()


async def send_to_backend(user_message: str, analysis: str) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            str(settings.backend_url),
            json={"message": user_message, "analysis": analysis},
        )


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
        analysis = await analyze_with_ollama(message.text)
        logging.info("Ollama analysis: %s", analysis)
    except Exception:
        logging.exception("Ollama request failed")
        analysis = message.text

    try:
        await send_to_backend(message.text, analysis)
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
