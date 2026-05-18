"""命令行交互式入口。"""

import json
import uuid
import asyncio

from .config import LLM_PROVIDER
from . import chat, clear_thread, get_history, get_service


async def main():
    """命令行交互式入口，支持多轮对话。"""
    thread_id = str(uuid.uuid4())[:8]

    print("=" * 60)
    print("  Agentic RAG v2 — 航天测控智能问答（意图路由版）")
    print(f"  LLM: {LLM_PROVIDER}  |  会话 ID: {thread_id}")
    print("  路径: chat闲聊 | sql数据库 | kb知识库 | web网络搜索")
    print("  输入 'quit' 退出 | 'clear' 清空对话 | 'history' 查看历史")
    print("=" * 60)

    print("  正在加载模型和索引...", flush=True)
    await get_service()
    print("  就绪，开始对话吧！\n")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "clear":
            await clear_thread(thread_id)
            print("[已清空对话历史]")
            continue
        if user_input.lower() == "history":
            history = await get_history(thread_id)
            if not history:
                print("（无对话历史）")
            else:
                for i, msg in enumerate(history):
                    role = "你" if msg["role"] == "user" else "Agent"
                    content = msg["content"][:200]
                    print(f"\n[{i+1}] {role}: {content}...")
            continue

        print("\n思考中...", flush=True)
        try:
            answer = await chat(user_input, thread_id)
            try:
                parsed = json.loads(answer)
                print(f"\n{json.dumps(parsed, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError:
                print(f"\n{answer}")
        except Exception as e:
            print(f"\n出错了: {e}")


if __name__ == "__main__":
    asyncio.run(main())
