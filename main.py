import asyncio
import json
import logging
import random
import sys
from collections import deque
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
MESSAGE_CONTEXT_LIMIT = 8
chat_message_history: dict[int, deque[str]] = {}


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


def get_previous_messages(chat_id: int) -> list[str]:
    return list(chat_message_history.get(chat_id, ()))


def remember_message(chat_id: int, text: str) -> None:
    history = chat_message_history.setdefault(
        chat_id,
        deque(maxlen=MESSAGE_CONTEXT_LIMIT),
    )
    history.append(text)


def format_previous_messages(messages: list[str]) -> str:
    if not messages:
        return "Истории пока нет."

    return "\n".join(
        f"{index}. {message}" for index, message in enumerate(messages, start=1)
    )


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


async def analyze_with_ollama(
    text: str,
    previous_messages: list[str],
) -> MessageContext:
    prompt = (
        "Проанализируй текущее сообщение пользователя и определи его намерение по смыслу всей переписки, "
        "а не по отдельным ключевым словам.\n"
        "Пользователи обычно пишут на русском языке. Учитывай разговорные русские формулировки, "
        "неполные фразы, местоимения и отсылки к предыдущим сообщениям. "
        "Не требуй точных слов «привет», «здравствуй», «пока» или «до свидания».\n"
        "Используй предыдущие сообщения только если текущее сообщение неполное или зависит от контекста. "
        "Например: «ему тоже», «и от меня», «скажи то же самое», «давай завершим», "
        "«передай то же», «можно от меня так же».\n"
        "Если по контексту пользователь хочет поздороваться, передать привет или попросить кого-то передать привет, "
        'intent должен быть "hello".\n'
        "Если по контексту пользователь хочет попрощаться, передать прощание, завершить разговор "
        "или попросить кого-то попрощаться, "
        'intent должен быть "bye".\n'
        'Если даже с учетом контекста намерение нельзя уверенно понять, intent должен быть "other".\n'
        "Верни только валидный JSON без Markdown и пояснений в формате:\n"
        '{"intent":"hello|bye|other","analysis":"краткая суть сообщения на русском"}\n\n'
        "Предыдущие сообщения этой переписки от старых к новым:\n"
        f"{format_previous_messages(previous_messages)}\n\n"
        f"Текущее сообщение: {text}"
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

    previous_messages = get_previous_messages(message.chat.id)

    try:
        context = await analyze_with_ollama(message.text, previous_messages)
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

    remember_message(message.chat.id, message.text)
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
