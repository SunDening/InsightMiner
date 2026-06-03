"""Auth API — register, login, email verification, profile."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, field_validator

import bcrypt

from insight_miner.config import JWT_ALGORITHM, JWT_EXPIRE_HOURS, JWT_SECRET
from insight_miner.core.store.database import DatabasePool
from insight_miner.utils.email import generate_code, send_verification_code

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)


# ── Password helpers (bcrypt) ──


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
security = HTTPBearer(auto_error=False)


# ── Pydantic schemas ──


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2 or len(v) > 32:
            raise ValueError("用户名长度需在 2-32 个字符之间")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("两次输入的密码不一致")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("密码长度不能少于 6 位")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class SendCodeRequest(BaseModel):
    email: str


class LoginWithCodeRequest(BaseModel):
    email: str
    code: str


class UserInfo(BaseModel):
    id: int
    username: str
    email: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo


class MessageResponse(BaseModel):
    message: str


# ── Helpers ──


def create_token(user_id: int, username: str, email: str) -> str:
    expire = datetime.now(UTC) + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "email": email,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserInfo:
    """Extract and verify the current user from the Bearer token."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub", 0))
        username = payload.get("username", "")
        email = payload.get("email", "")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的令牌")
        return UserInfo(id=user_id, username=username, email=email)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的令牌")


# ── Routes ──


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """Register a new user."""
    pool = await DatabasePool.get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="数据库不可用")

    async with pool.acquire() as conn:
        # Check existing
        row = await conn.fetchrow("SELECT id FROM users WHERE email = $1", req.email)
        if row:
            raise HTTPException(status_code=409, detail="该邮箱已被注册")

        row = await conn.fetchrow("SELECT id FROM users WHERE username = $1", req.username)
        if row:
            raise HTTPException(status_code=409, detail="该用户名已被使用")

        hashed = hash_password(req.password)
        user = await conn.fetchrow(
            "INSERT INTO users (username, email, password) VALUES ($1, $2, $3) RETURNING id, username, email",
            req.username,
            req.email,
            hashed,
        )

    user_info = UserInfo(id=user["id"], username=user["username"], email=user["email"])
    token = create_token(user_info.id, user_info.username, user_info.email)
    return AuthResponse(access_token=token, user=user_info)


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Login with email and password."""
    pool = await DatabasePool.get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="数据库不可用")

    async with pool.acquire() as conn:
        user = await conn.fetchrow("SELECT id, username, email, password FROM users WHERE email = $1", req.email)

    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")

    user_info = UserInfo(id=user["id"], username=user["username"], email=user["email"])
    token = create_token(user_info.id, user_info.username, user_info.email)
    return AuthResponse(access_token=token, user=user_info)


@router.post("/send-code", response_model=MessageResponse)
async def send_code(req: SendCodeRequest):
    """Send a verification code to the user's email."""
    pool = await DatabasePool.get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="数据库不可用")

    code = generate_code()
    expires_at = datetime.now(UTC) + timedelta(minutes=5)
    logger.info("Generated code %s for %s", code, req.email)

    async with pool.acquire() as conn:
        # Invalidate old unused codes
        await conn.execute(
            "UPDATE verification_codes SET used = TRUE WHERE email = $1 AND purpose = 'login' AND used = FALSE",
            req.email,
        )
        await conn.execute(
            "INSERT INTO verification_codes (email, code, purpose, expires_at) VALUES ($1, $2, 'login', $3)",
            req.email,
            code,
            expires_at,
        )

    ok = await send_verification_code(req.email, code)
    if not ok:
        raise HTTPException(status_code=500, detail="验证码发送失败，请稍后重试")

    return MessageResponse(message="验证码已发送")


@router.post("/login-with-code", response_model=AuthResponse)
async def login_with_code(req: LoginWithCodeRequest):
    """Login with email + verification code."""
    pool = await DatabasePool.get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="数据库不可用")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id FROM verification_codes
               WHERE email = $1 AND code = $2 AND purpose = 'login'
                 AND used = FALSE AND expires_at > NOW()""",
            req.email,
            req.code,
        )
        if not row:
            raise HTTPException(status_code=401, detail="验证码无效或已过期")

        # Mark code used
        await conn.execute(
            "UPDATE verification_codes SET used = TRUE WHERE id = $1", row["id"]
        )

        # Find or create user
        user = await conn.fetchrow("SELECT id, username, email FROM users WHERE email = $1", req.email)
        if not user:
            # Auto-create user with email as username prefix
            username = req.email.split("@")[0]
            # Ensure unique
            existing = await conn.fetchrow("SELECT id FROM users WHERE username = $1", username)
            if existing:
                username = f"{username}_{datetime.now(UTC).strftime('%H%M%S')}"
            dummy_password = hash_password("")  # no password login allowed
            user = await conn.fetchrow(
                "INSERT INTO users (username, email, password) VALUES ($1, $2, $3) RETURNING id, username, email",
                username,
                req.email,
                dummy_password,
            )

    user_info = UserInfo(id=user["id"], username=user["username"], email=user["email"])
    token = create_token(user_info.id, user_info.username, user_info.email)
    return AuthResponse(access_token=token, user=user_info)


@router.get("/me", response_model=UserInfo)
async def get_me(current_user: UserInfo = Depends(get_current_user)):
    """Return the current logged-in user's info."""
    return current_user
