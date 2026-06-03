"""Email sending utility via QQ SMTP (SSL)."""

from __future__ import annotations

import logging
import random
import string

from aiosmtplib import SMTP

from insight_miner.config import EMAIL_FROM, EMAIL_HOST, EMAIL_PASSWORD, EMAIL_PORT, EMAIL_USER

logger = logging.getLogger(__name__)


def generate_code(length: int = 6) -> str:
    """Generate a numeric verification code."""
    return "".join(random.choices(string.digits, k=length))


async def send_verification_code(to_email: str, code: str) -> bool:
    """Send a verification code email via QQ SMTP."""
    subject = "InsightMiner 登录验证码"
    body = (
        f"您的验证码是：{code}\n\n"
        f"该验证码 5 分钟内有效，请勿泄露给他人。\n\n"
        f"—— InsightMiner 团队"
    )

    message = (
        f"From: {EMAIL_FROM}\n"
        f"To: {to_email}\n"
        f"Subject: {subject}\n"
        f"Content-Type: text/plain; charset=utf-8\n\n"
        f"{body}"
    )

    try:
        smtp = SMTP(hostname=EMAIL_HOST, port=EMAIL_PORT, use_tls=True)
        await smtp.connect()
        await smtp.login(EMAIL_USER, EMAIL_PASSWORD)
        await smtp.sendmail(EMAIL_FROM, [to_email], message)
        await smtp.quit()
        logger.info("Verification code sent to %s", to_email)
        return True
    except Exception as e:
        logger.error("Failed to send email to %s: %s", to_email, e)
        return False
