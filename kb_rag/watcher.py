"""KB 目录事件监控 — 基于 watchdog（ReadDirectoryChangesW / inotify / FSEvents）。

订阅 OS 文件系统事件，debounce 去重后触发 RagKnowledgeBase 增量更新。

与轮询方案比：
  - 事件驱动，无轮询延迟
  - 由 watchdog 封装各平台原生 API（Win: ReadDirectoryChangesW, Linux: inotify, macOS: FSEvents）
"""

import os
import threading
import asyncio
import logging

from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from .config import _SUPPORTED_EXTS

logger = logging.getLogger("kb_rag.watcher")


class _KbEventHandler(FileSystemEventHandler):
    """watchdog 事件处理器 — 归并 KB 文件变更后批量回调。"""

    def __init__(self, kb, loop=None, debounce_sec: float = 2.0):
        self.kb = kb
        self.loop = loop
        self.debounce_sec = debounce_sec
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._pending = False

    def on_created(self, event: FileSystemEvent) -> None:
        self._debounce(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._debounce(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._debounce(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        # 移动目标文件可能出现在 KB_DIR 外，统一起伏重新扫描
        self._debounce(event)

    # ── 内部 ────────────────────────────────────────────────

    def _is_supported(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in _SUPPORTED_EXTS

    def _debounce(self, event: FileSystemEvent) -> None:
        """对每个文件事件去重：2 秒内同一触发窗口合并为一次更新。"""
        if event.is_directory:
            return
        if not self._is_supported(event.src_path):
            return

        with self._lock:
            if self._pending:
                return
            self._pending = True
            self._timer = threading.Timer(self.debounce_sec, self._apply)
            self._timer.daemon = True
            self._timer.start()

    def _apply(self) -> None:
        """执行增量更新快照。"""
        with self._lock:
            self._pending = False

        try:
            changes = self.kb._detect_file_changes()
            if not any(changes.values()):
                return

            logger.info(
                f"[Watcher] 检测到变更: +{len(changes['new'])} "
                f"~{len(changes['modified'])} -{len(changes['deleted'])}"
            )
            self.kb._apply_file_changes(changes)

            # 摘要刷新跑在主事件循环上
            if self.loop is not None and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.kb._refresh_descriptions(), self.loop
                )

            self.kb._finalize_index()
        except Exception as e:
            logger.error(f"[Watcher] 更新失败: {e}")


class KbDirWatcher:
    """watchdog Observer 封装 — start/stop 生命周期。"""

    def __init__(self, kb, loop=None, kb_dir: str = "",
                 debounce_sec: float = 2.0):
        self._observer = Observer()
        handler = _KbEventHandler(kb, loop=loop, debounce_sec=debounce_sec)
        self._observer.schedule(handler, kb_dir, recursive=False)
        self._observer.daemon = True

    def start(self) -> None:
        """启动 watchdog Observer（自带独立线程）。"""
        self._observer.start()
        logger.info("[Watcher] 已启动（watchdog 事件驱动）")

    def stop(self) -> None:
        """停止 Observer 并等待线程退出。"""
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join(timeout=5)
        logger.info("[Watcher] 已停止")
