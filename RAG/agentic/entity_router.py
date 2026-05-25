"""Entity Router — 业务实体路由。

定义 6 个业务实体，查询先匹配实体，再在实体范围内检索——将 64 表搜索空间压缩为 7-13 表。
路由方式：BM25 关键词匹配 + 实体描述语义匹配。
"""

import re

from .config import logger


class EntityRouter:
    """业务实体路由器。

    基于 table_desc.json 的模块划分，定义 6 个业务实体。
    每个实体包含其业务关键词、描述和所属表列表。
    """

    # 实体定义（来自 table_desc.json 的模块 + 手工调整）
    ENTITIES = {
        "卫星网络通知": {
            "tables": [
                "notice", "com_el", "adm_assoc", "ntc_memo", "attch",
                "ntc_commit",
            ],
            "keywords": [
                "通知", "notice", "提交", "主管部门", "国家代码", "adm",
                "接收日期", "ntc_id", "通函", "公布", "附文",
            ],
            "description": "卫星网络通知的提交、审批、公布流程相关数据",
        },
        "频率指配与波束": {
            "tables": [
                "grp", "carrier_fr", "s_beam", "srv_area", "srv_cls",
                "strap", "mod_char", "mask_info", "mask_lnk1", "mask_lnk2",
                "ngma", "pwr_ctrl",
            ],
            "keywords": [
                "频率", "freq", "指配", "波束", "beam", "载波", "carrier",
                "GHz", "MHz", "kHz", "频段", "band", "EIRP", "PFD", "掩码",
                "调制", "modulation", "上行", "下行", "uplink", "downlink",
                "增益", "gain", "极化", "业务区", "服务区",
            ],
            "description": "频率指配组、波束参数、发射特性、调制参数、掩码等频率相关数据",
        },
        "空间电台与轨道": {
            "tables": [
                "geo", "non_geo", "orbit", "phase", "orbit_lnk",
                "sat_oper", "c_pfd",
            ],
            "keywords": [
                "GSO", "对地静止", "geostationary", "NGSO", "非对地静止",
                "轨道", "orbit", "倾角", "inclination", "远地点", "近地点",
                "轨道面", "卫星", "satellite", "空间电台", "space station",
                "相位", "phase", "PFD限值",
            ],
            "description": "GSO/NGSO 空间电台的轨道位置、轨道平面参数相关数据",
        },
        "地球站": {
            "tables": [
                "e_stn", "e_ant", "e_ant_elev", "hor_elev",
                "e_as_stn", "e_srvcls",
            ],
            "keywords": [
                "地球站", "earth station", "天线", "antenna", "仰角",
                "elevation", "经纬度", "波束宽度", "增益图",
                "辐射方向图", "水平仰角",
            ],
            "description": "地球站位置、天线参数、仰角限制、地形数据等",
        },
        "商业系统与航天器": {
            "tables": [
                "cmr_syst", "cmr_grp_lnk", "cmr_notice",
                "scraft_cmr_syst", "scraft_cmr_freq", "cmr_history",
            ],
            "keywords": [
                "商业", "commercial", "制造商", "发射商", "航天器",
                "spacecraft", "卫星", "satellite", "SRS",
            ],
            "description": "商业卫星系统、航天器信息、发射历史等",
        },
        "干扰计算与结果": {
            "tables": [
                "ap30b_tr_res", "ap30b_ref_se", "ap30b_ref_agg",
                "link_epm", "ovrl_epm", "sps_results",
                "tr_aff_ntw", "tr_provn", "sat_sys_provn",
            ],
            "keywords": [
                "干扰", "interference", "C/I", "保护余量", "EPM",
                "附录30", "Appendix 30", "受影响网络", "协调",
                "coordination", "provn",
            ],
            "description": "附录30/30A/30B 干扰计算结果、受影响网络、协调数据",
        },
    }

    # 全局参照表（不归属特定实体，所有查询都可访问）
    GLOBAL_TABLES = {
        "alloc_id", "ant_type", "beam_tr", "cost_recov", "diag_grp",
        "ex_op_grp", "fdg_ref", "grp_lnk", "ntc_lnk", "ntc_lnk_ref",
        "plan_pub", "pub_ssn", "res49_sel", "srs_ooak", "strap",
        "pl_strap", "sat_lnk", "s_as_stn", "cmr_history", "coord_agree_ntw",
    }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中英混合分词。"""
        tokens = []
        for m in re.finditer(r"[a-zA-Z0-9]+", text.lower()):
            tokens.append(m.group())
        for ch in re.findall(r"[一-鿿]", text):
            tokens.append(ch)
        for i in range(len(tokens) - 1):
            if re.match(r'[一-鿿]', tokens[i]) and re.match(r'[一-鿿]', tokens[i + 1]):
                tokens.append(tokens[i] + tokens[i + 1])
        return tokens

    def route(self, query: str, max_entities: int = 2) -> list[str]:
        """将查询路由到最匹配的 1-2 个实体。

        Returns:
            实体内表名的并集列表（去重）。
        """
        query_tokens = self._tokenize(query)
        entity_scores: dict[str, float] = {}

        for entity_name, entity_info in self.ENTITIES.items():
            score = 0.0
            for kw in entity_info["keywords"]:
                kw_lower = kw.lower()
                kw_tokens = self._tokenize(kw)
                for qt in query_tokens:
                    if qt == kw_lower or qt in kw_tokens or kw_lower in qt:
                        score += 2.0  # 精确或部分匹配
                    elif len(qt) >= 2 and len(kw_lower) >= 2:
                        # 子串匹配
                        if qt in kw_lower or kw_lower in qt:
                            score += 1.0
            entity_scores[entity_name] = score

        # 按分数降序
        ranked = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)

        # 取 top-N 实体（分数 > 0）
        selected_entities = [
            name for name, score in ranked[:max_entities] if score > 0
        ]
        if not selected_entities:
            # 无明确匹配 → 返回全球表（最常见的通知 + 频率实体）
            selected_entities = ["卫星网络通知"]

        # 汇总表集合
        table_set = set()
        for entity_name in selected_entities:
            table_set.update(self.ENTITIES[entity_name]["tables"])
        # 加上全局参照表
        table_set.update(self.GLOBAL_TABLES)

        logger.info(
            f"[EntityRouter] query → {selected_entities} "
            f"({len(table_set)} 张表)"
        )
        return list(table_set)
