# Signos Ego-Network 数据

巴塞罗那移民个人网络调查（**SIGnos**），由 EgoLab 实施、Fundació ACSAR 资助。数据以 EgoNet 软件原生 XML 格式发布，本仓库将其转换为可读的 JSON 三表，便于与 GSS 等其他 ego-network 数据集对齐分析。

- **官方说明**：[visone — Signos (data)](https://visone.ethz.ch/wiki/index.php/Signos_(data))
- **相关书目**：Jose Luis Molina & Fabien Pelissier (eds.) (2010). *Les xarxes socials de sikhs, xinesos i filipins a Barcelona*. Barcelona: Fundació ACSAR.
- **更多个人网络数据**：[OPeN — Observatory of Personal Networks](https://www.open-personal-networks.org/)

---

## 快速开始

```python
import json
from pathlib import Path

ROOT = Path("Survey/signos/processed")

# 汇总表：70 人 / 2100 alter / 30450 边
ego   = json.loads((ROOT / "ego.json").read_text())
alter = json.loads((ROOT / "alter.json").read_text())
ties  = json.loads((ROOT / "alter_alter.json").read_text())

# 逐人可读版本（推荐初次浏览）
one = json.loads((ROOT / "by_interview/chinese/chinese0.json").read_text())
print(one["ego"]["age"]["label"])          # "20"
print(one["alters"][0]["closeness"]["label"])  # "Muy próximo_a"
```

```bash
# 从原始 XML 重新生成 JSON
python3 Survey/signos/scripts/export_json.py
```

---

## 样本概览

| 项目 | 内容 |
|------|------|
| **地区** | 西班牙巴塞罗那 |
| **族群** | 华人 (chinese) 21 人、菲律宾人 (filipinos) 25 人、锡克教徒 (sikhs) 24 人 |
| **总样本** | 70 个 ego-centric 网络访谈 |
| **网络界定** | 每位 Ego 提名最多 **30** 个 Alter |
| **Alter–Alter 结构** | 完整无向对（C(30,2) = **435** 对 / 人） |
| **纵向** | 否（横截面，约 2009 年） |
| **完成率** | 70 / 70（`Complete=true`） |

问卷版本：62 份使用正式问卷（`signos.ego`），8 份使用试点问卷（`signos_p_piloto.ego`）。试点访谈：chinese 4 份、filipinos 3 份、sikhs 1 份。

---

## 目录结构

```
Survey/signos/
├── README.md
├── signos.ego                 # 正式问卷定义（32 题，XML）
├── signos_p_piloto.ego        # 试点问卷定义（XML）
├── interviews/                # 原始访谈回答（XML）
│   ├── chinese/   (*.int × 21)
│   ├── filipinos/ (*.int × 25)
│   └── sikhs/     (*.int × 24)
├── scripts/
│   └── export_json.py         # XML → JSON 转换脚本
└── processed/                 # 派生 JSON（推荐使用）
    ├── manifest.json
    ├── questionnaire.json     # 问卷题目与选项定义
    ├── variable_labels.json   # 变量名 → 西班牙语题号
    ├── value_labels.json      # 选择题取值码说明
    ├── ego.json               # 70 行，wide
    ├── alter.json             # 2100 行，long
    ├── alter_alter.json       # 30450 行，边列表
    └── by_interview/          # 逐人嵌套 JSON
        ├── chinese/
        ├── filipinos/
        └── sikhs/
```

原始 XML 保留不动；`processed/` 可随时用脚本重跑覆盖。

---

## 表关系

```
ego                              alter
┌────────────────────────┐      ┌─────────────────────────────┐
│ ego_id  (PK)           │─1:N─►│ ego_id + alter_id  (复合键) │
│ community, age, sex…     │      │ sex, closeness, relation…   │
└────────────────────────┘      └─────────────────────────────┘
         │
         │              alter_alter
         └──── 1:N ────► ego_id + source_id + target_id
                          value, probably_knows
```

- 每位 Ego 固定 **30** 个 Alter（`alter_id` = `"0"` … `"29"`）
- 每位 Ego 有 **435** 条 Alter–Alter 对（所有无序对）
- `ego_id` 格式：`{community}{编号}`，如 `chinese0`、`filipinos12`、`sikhs5`

---

## processed/ 文件说明

| 文件 | 行数 / 粒度 | 用途 |
|------|-------------|------|
| `ego.json` | 70 行，一人一行 | 批量分析 Ego 人口统计与移民变量 |
| `alter.json` | 2100 行，一 Alter 一行 | Ego–Alter 关系与 Alter 属性 |
| `alter_alter.json` | 30450 行，一边一行 | 个人网络内部结构 |
| `by_interview/{community}/{ego_id}.json` | 70 个文件 | 人工阅读、检查个案 |
| `questionnaire.json` | 2 套问卷 | 查题目原文与选项编码 |
| `value_labels.json` | 按变量 | 查选择题 `value` 含义 |
| `variable_labels.json` | 按变量 | 查变量对应的西班牙语题号 |

### 数据类型约定

| 类型 | 说明 |
|------|------|
| `string` | 文本，如 `ego_id`、`community`、`*_label` |
| `integer` | 编码值 `value`；开放文本题也可能存为整数内部码 |
| `boolean` | 如 `complete`、`probably_knows` |
| `null` | 跳题未答、或该题不适用 |

汇总表（`ego.json` 等）中，每个变量通常有 `变量名`（编码）和 `变量名_label`（显示文本）两列。

---

## 字段说明

完整取值码见 `processed/value_labels.json`；题号标题见 `variable_labels.json`。

### ego（22 题）

| 变量 | 西班牙语题号 | 含义 |
|------|-------------|------|
| `sex` | 1. Hombre-mujer | 性别 |
| `age` | 2. Edad | 年龄 |
| `country` | 3. País | 本人国籍 / 来源国 |
| `religion` | 4. Religión | 宗教 |
| `migration_reason` | 5. Razones | 来西班牙的原因 |
| `father_country` | 6. País padre | 父亲出生国 |
| `mother_country` | 7. País madre | 母亲出生国 |
| `marital_status` | 8. Estado civil | 婚姻状况 |
| `num_children` | 9. Hijos_as | 子女数 |
| `employment_status` | 10. Situación laboral | 就业状况 |
| `legal_status` | 11. Regularización | 居留是否合法 |
| `occupation` | 12. Actividad | 职业（开放文本） |
| `education` | 13. Estudios | 教育程度 |
| `years_residence` | 14. Tiempo residencia | 在西班牙居住年数 |
| `visits_origin` | 15. Visitas | 回访原籍国次数 |
| `eu_member` | 16. Unión Europea | 是否欧盟成员国公民 |
| `eu_countries` | 17. Países de la Unión Europea | 具体欧盟国家（开放文本） |
| `remittances` | 18. Remesas | 是否向原籍国汇款 |
| `remittance_use` | 19. Uso remesas | 汇款用途（建房/生意等） |
| `family_in_origin` | 20. Familia en origen | 原籍国是否还有家人 |
| `remittance_recipient` | 21. Receptor remesas | 汇款接收人 |
| `gifts_to_origin` | 22. Regalos origen | 向原籍国寄送礼物次数 |

**跳题逻辑**：若 `remittances = "Nunca"`（从不汇款），则 `remittance_use`、`family_in_origin`、`remittance_recipient`、`gifts_to_origin` 为 `null`（约 50+ 人）。若 `eu_member = "No"`，则 `eu_countries` 为 `null`。

### alter（8 题 × 30 人）

| 变量 | 西班牙语题号 | 含义 |
|------|-------------|------|
| `sex` | 24. Alter-hombre-mujer | Alter 性别 |
| `closeness` | 25. Proximidad | 与 Ego 亲近程度（0–4） |
| `alter_country` | 26. País alter | Alter 来源国 |
| `alter_residence_country` | 27. Residencia alter | Alter 居住国（开放文本） |
| `alter_residence_city` | 28. Localidad residencia | Alter 居住城市（开放文本） |
| `confidant` | 29. Confidencia | 是否向 Alter 倾诉 |
| `relation_type` | 30. Razón encuentro | 与 Ego 的关系类型 |
| `meeting_place` | 31. Lugar de encuentro | 常见见面地点（巴塞罗那地铁站点，开放文本） |

**注意**：问卷中 **没有 Alter 年龄题**；Alter 侧可用的人口统计变量为性别、来源国、居住国/城市及关系类型。

### alter_alter（Alter 间关系）

每行表示一对 Alter 是否可能互相认识（由 Ego 判断）：

| 字段 | 含义 |
|------|------|
| `source_id`, `target_id` | Alter 编号（`"0"`–`"29"`） |
| `value` | 0 = 不太可能，1 = 有可能，2 = 很可能 |
| `label` | 西班牙语回答文本 |
| `probably_knows` | `value == 2` 时为 `true`（派生字段，便于筛边） |

三档回答分布（70 人 × 435 对 = 30,450）：

| 回答 | 比例 |
|------|------|
| No es probable | 60.6% |
| Muy probablemente | 31.9% |
| Podría ser | 7.5% |

分析网络结构时，通常将 `probably_knows = true` 视为有边；`Podría ser` 是否纳入需根据研究设定自行决定。

---

## by_interview/ 嵌套 JSON 结构

每个文件对应一位 Ego 的完整访谈，结构比汇总表更易读：

```json
{
  "ego_id": "chinese0",
  "community": "chinese",
  "questionnaire": "pilot",
  "complete": true,
  "num_alters": 30,
  "alter_ids": ["0", "1", "...", "29"],
  "ego": {
    "sex": {"value": 1, "label": "Hombre"},
    "age": {"value": 20, "label": "20"},
    "remittance_use": null
  },
  "alters": [
    {
      "alter_id": "0",
      "sex": {"value": 0, "label": "Hombre"},
      "closeness": {"value": 4, "label": "Muy próximo_a"}
    }
  ],
  "alter_alter_ties": [
    {
      "source_id": "0",
      "target_id": "1",
      "value": 2,
      "label": "Muy probablemente",
      "probably_knows": true
    }
  ]
}
```

---

## 原始 XML 格式（.int / .ego）

若需对照原始数据，每个 `.int` 文件结构如下：

```xml
<Interview StudyId="..." StudyName="SIGnos" NumAlters="30">
  <AlterList><Name>0</Name>…<Name>29</Name></AlterList>
  <AnswerList>
    <Answer>
      <QuestionId>...</QuestionId>
      <Answered>true</Answered>
      <Value>...</Value>
      <String>...</String>
      <Alters><Index name="0">0</Index></Alters>  <!-- Alter 题：1 个 Index -->
    </Answer>
    <!-- Alter–Alter 题：<Alters> 含 2 个 <Index> -->
  </AnswerList>
  <EgoName><First>chinese0</First></EgoName>
  <Complete>true</Complete>
</Interview>
```

- `.ego`：问卷定义（题目文本、选项列表）；`.int` 通过 `QuestionId` 引用 `.ego`
- `QuestionType`：1 = Ego，2 = 提名，3 = Alter，4 = Alter–Alter
- 公开发布版已将真实姓名替换为数字 ID

---

## 数据流

```
Signos_public_data.zip（官方）
        │
        ▼
signos.ego + interviews/*/*.int   ← 原始 XML（本仓库保留）
        │
        │  python3 Survey/signos/scripts/export_json.py
        ▼
processed/*.json                  ← 可读三表 + 元数据
```

---

## 与 GSS 的对比

| 项目 | Signos | GSS（如 1985） |
|------|--------|----------------|
| 样本量 | 70 egos | ~1,500 egos |
| 每 ego 最多 alter 数 | 30 | 5 |
| Alter–Alter 边 | 有（概率三档） | 有（close 矩阵，仅 1985/2004） |
| Alter 年龄 | 无 | 有 |
| 导出格式 | JSON | CSV + Parquet |
| 分析入口 | 直接读 JSON | `GSS.scripts.load_gss.load_egor_year()` |

---

## 注意事项

1. **匿名化**：Alter 真实姓名已替换为 0–29；提名题仅记录人数，不保留原名。
2. **试点问卷**：8 份试点访谈在部分选项措辞上与正式版略有差异（如关系类型、Alter 来源国选项），跨问卷比较时建议查 `questionnaire.json` 中 `main` / `pilot` 两套定义。
3. **开放文本题**：`occupation`、`alter_residence_city`、`meeting_place` 等 `*_label` 为受访者原文，大小写与拼写不统一。
4. **边定义**：`probably_knows` 仅为一种常用二值化方案，非官方唯一标准。

---

*最后更新：2025-06-29*
