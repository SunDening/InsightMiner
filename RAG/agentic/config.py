"""Agentic RAG v2 — 配置常量、环境变量、日志系统、Schema 定义。

所有路径均相对于 RAG/ 项目根目录。
"""

import os
import re
import json
import logging
import threading
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

# ── 项目根目录 ────────────────────────────────────────────────────────
# config.py 位于 RAG/agentic/，上溯两级即 RAG/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 日志（每 10 秒批量写入文件，每次启动清空）──────────────────────────
LOG_FILE = os.path.join(PROJECT_ROOT, "agentic_rag.log")
LOG_INTERVAL = 10

# 启动时清空旧日志
with open(LOG_FILE, "w", encoding="utf-8") as _f:
    _f.write("")

_log_buffer: list[str] = []
_log_lock = threading.Lock()


def _flush_logs():
    with _log_lock:
        if _log_buffer:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write("\n".join(_log_buffer) + "\n")
            _log_buffer.clear()


class _TimedBufferingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _log_lock:
                _log_buffer.append(msg)
        except Exception:
            self.handleError(record)


def _start_log_timer():
    def _loop():
        while True:
            _flush_logs()
            threading.Event().wait(LOG_INTERVAL)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


_start_log_timer()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[_TimedBufferingHandler()],
)
logger = logging.getLogger("agentic_rag_v2")

# ── MySQL ──────────────────────────────────────────────────────────────

MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER", "knight"),
    "password": os.getenv("MYSQL_PASSWORD", "123456"),
    "database": os.getenv("MYSQL_DATABASE", "test_data"),
}

# ── LLM ────────────────────────────────────────────────────────────────

LLM_PROVIDER: Literal["deepseek", "ollama"] = "ollama"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

MAX_SQL_RETRY = 3
MAX_REVIEW_RETRY = 2

# ── 知识库路径 ─────────────────────────────────────────────────────────

KB_DIR = os.path.join(PROJECT_ROOT, "kb")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
SUMMARY_CACHE_PATH = os.path.join(PROJECT_ROOT, ".kb_summaries.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── 支持的知识库文档格式 ───────────────────────────────────────────────

_SUPPORTED_EXTS: set[str] = {".txt", ".md", ".csv", ".json", ".yaml", ".pdf", ".docx", ".doc"}


# ── 危险 SQL 关键词黑名单 ──────────────────────────────────────────────

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "RENAME", "LOAD",
    "IMPORT", "EXEC", "EXECUTE", "CALL", "MERGE",
]


# ── 全量 Schema 定义 ────────────────────────────────────────────────────

SCHEMA_DEFINITION = r"""
你拥有一个 MySQL 8.0 数据库 test_data，其中包含以下 8 张表。
请仔细阅读表结构、字段含义和外键关系，然后根据用户问题生成精确的 SQL 查询。

────────────────────────────────────────────────────────────────────────
表1：launch_vehicles（运载火箭表）
────────────────────────────────────────────────────────────────────────
  vehicle_id        INT          主键，自增
  vehicle_code      VARCHAR      火箭编号（如 CZ-2F、CZ-5B、Falcon-9）
  provider_name     VARCHAR      制造方/提供方（如 CASC、SpaceX）
  first_stage_type  VARCHAR      一级推进剂或构型（如 Kerosene、LOX/LH2）
  lift_capacity_kg  INT          近地轨道运力（LEO），单位 kg
  recovery_mode     VARCHAR      回收方式（Expendable、Propulsive、Parachute）
  关联：vehicle_id → launch_missions.vehicle_id（1对N）

────────────────────────────────────────────────────────────────────────
表2：launch_missions（发射任务表）
────────────────────────────────────────────────────────────────────────
  mission_id        INT          主键，自增
  vehicle_id        INT          外键 → launch_vehicles.vehicle_id
  mission_code      VARCHAR      任务编号
  launch_site       VARCHAR      发射场（如 Jiuquan、Xichang、Wenchang）
  launch_time       DATETIME     发射时间（UTC）
  target_orbit      VARCHAR      目标轨道（LEO、MEO、GEO、SSO、GTO）
  mission_status    VARCHAR      发射结果（Success、Failure、Partial Failure）
  关联：vehicle_id → launch_vehicles.vehicle_id
        mission_id → satellites.mission_id / tracking_sessions.mission_id（1对N）

────────────────────────────────────────────────────────────────────────
表3：satellites（卫星表）
────────────────────────────────────────────────────────────────────────
  satellite_id      INT          主键，自增
  mission_id        INT          外键 → launch_missions.mission_id
  satellite_code    VARCHAR      卫星编号（如 SAT-001）
  orbit_plane       VARCHAR      轨道面或轨位
  nominal_altitude_km INT        标称轨道高度，单位 km
  primary_band      VARCHAR      主业务频段（C、Ku、Ka、X、S、UHF、L）
  service_status    VARCHAR      服役状态（Active、Retired、Backup、Testing）
  entered_service   DATETIME     入轨/入网时间
  关联：mission_id → launch_missions.mission_id
        satellite_id → frequency_allocations.satellite_id / tracking_sessions.satellite_id（1对N）

────────────────────────────────────────────────────────────────────────
表4：ground_stations（地面站表）
────────────────────────────────────────────────────────────────────────
  station_id        INT          主键，自增
  station_code      VARCHAR      地面站编号
  station_name      VARCHAR      地面站名称
  country_region    VARCHAR      所在区域
  uplink_band       VARCHAR      常用上行频段
  commissioned_on   DATE         投运日期
  关联：station_id → frequency_allocations.station_id / tracking_sessions.station_id（1对N）

────────────────────────────────────────────────────────────────────────
表5：frequency_allocations（频率指配表，卫星↔地面站桥表）
────────────────────────────────────────────────────────────────────────
  allocation_id     INT          主键，自增
  satellite_id      INT          外键 → satellites.satellite_id
  station_id        INT          外键 → ground_stations.station_id
  channel_label     VARCHAR      信道标签
  center_freq_mhz   DECIMAL      中心频率 MHz
  bandwidth_mhz     DECIMAL      带宽 MHz
  valid_from        DATETIME     生效时间
  link_role         VARCHAR      链路用途（Uplink、Downlink、TT&C Uplink、TT&C Downlink）
  说明：实现 satellites 与 ground_stations 的多对多关系

────────────────────────────────────────────────────────────────────────
表6：tracking_sessions（测控会话表，核心事实表）
────────────────────────────────────────────────────────────────────────
  session_id        INT          主键，自增
  mission_id        INT          外键 → launch_missions.mission_id
  satellite_id      INT          外键 → satellites.satellite_id
  station_id        INT          外键 → ground_stations.station_id
  session_start     DATETIME     会话开始时间（UTC）
  duration_min      INT          持续时长，分钟
  data_volume_gb    DECIMAL      下传/回传数据量 GB
  session_state     VARCHAR      会话状态（Completed、Interrupted、In Progress、Scheduled）
  关联：同时关联 mission、satellite、station 三张表

────────────────────────────────────────────────────────────────────────
表7：electromagnetic_events（电磁活动事件表）
────────────────────────────────────────────────────────────────────────
  event_id          INT          主键，自增
  source_sector     VARCHAR      事件来源扇区/区域
  band_label        VARCHAR      影响频段（C-band、Ku-band、X-band 等）
  event_time        DATETIME     事件发生时间（UTC）
  duration_sec      INT          持续秒数
  peak_intensity_dbuv DECIMAL    峰值强度 dBμV
  suspected_source  VARCHAR      疑似来源
  说明：独立表，通过 band_label / event_time / source_sector 与其他表隐式关联

────────────────────────────────────────────────────────────────────────
表8：space_weather_bulletins（空间天气通报表）
────────────────────────────────────────────────────────────────────────
  bulletin_id       INT          主键，自增
  bulletin_code     VARCHAR      通报编号
  bulletin_time     DATETIME     通报时间（UTC）
  solar_flux_sfu    INT          太阳射电流量 SFU
  geomagnetic_index_kp DECIMAL   地磁 Kp 指数（0-9）
  proton_flux_pfu   INT          质子通量 PFU
  advisory_level    VARCHAR      通报级别（Quiet、Watch、Warning、Severe）
  affected_orbit    VARCHAR      主要受影响轨道（LEO、MEO、GEO、ALL）
  说明：独立表，通过 bulletin_time / affected_orbit 做风险联动分析

────────────────────────────────────────────────────────────────────────
重要提示：
- 时间字段均为 UTC
- mission_status：Success / Failure / Partial Failure
- service_status：Active / Retired / Backup / Testing
- 字符串比较优先用 = 精确匹配，模糊搜索用 LIKE '%keyword%'
- 多表用 JOIN ... ON 并指定正确外键
- 聚合查询正确使用 GROUP BY 和聚合函数
- MySQL 8.0 语法，默认 LIMIT 50
"""
