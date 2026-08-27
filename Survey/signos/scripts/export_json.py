#!/usr/bin/env python3
"""Export Signos EgoNet XML (.ego / .int) to readable JSON tables."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIGNOS_DIR = SCRIPT_DIR.parent
INTERVIEWS_DIR = SIGNOS_DIR / "interviews"
PROCESSED_DIR = SIGNOS_DIR / "processed"

STUDY_MAIN = "1235423217375"
STUDY_PILOT = "1233609929687"

QUESTION_VARS = {
    "1233045267984": "sex",
    "1233045918046": "age",
    "1233045938375": "country",
    "1233046068421": "religion",
    "1233046319734": "migration_reason",
    "1233046536890": "father_country",
    "1233046660015": "mother_country",
    "1233046678921": "marital_status",
    "1233046773875": "num_children",
    "1233046874093": "employment_status",
    "1233047018125": "legal_status",
    "1233047085359": "occupation",
    "1233047093625": "education",
    "1233047253234": "years_residence",
    "1233047292203": "visits_origin",
    "1233047305625": "eu_member",
    "1233131650265": "eu_countries",
    "1233047345796": "remittances",
    "1233047406328": "remittance_use",
    "1233047518031": "family_in_origin",
    "1233047563390": "remittance_recipient",
    "1233047668968": "gifts_to_origin",
    "1233047986015": "alter_name_generator",
    "1233048009250": "sex",
    "1233048140187": "closeness",
    "1233048247421": "alter_country",
    "1233048585015": "alter_residence_country",
    "1233048601125": "alter_residence_city",
    "1233048625531": "confidant",
    "1233048420484": "relation_type",
    "1233048734703": "meeting_place",
    "1233048852562": "tie_probability",
}

QUESTION_TYPES = {
    "1233045267984": "ego",
    "1233045918046": "ego",
    "1233045938375": "ego",
    "1233046068421": "ego",
    "1233046319734": "ego",
    "1233046536890": "ego",
    "1233046660015": "ego",
    "1233046678921": "ego",
    "1233046773875": "ego",
    "1233046874093": "ego",
    "1233047018125": "ego",
    "1233047085359": "ego",
    "1233047093625": "ego",
    "1233047253234": "ego",
    "1233047292203": "ego",
    "1233047305625": "ego",
    "1233131650265": "ego",
    "1233047345796": "ego",
    "1233047406328": "ego",
    "1233047518031": "ego",
    "1233047563390": "ego",
    "1233047668968": "ego",
    "1233047986015": "alter_prompt",
    "1233048009250": "alter",
    "1233048140187": "alter",
    "1233048247421": "alter",
    "1233048585015": "alter",
    "1233048601125": "alter",
    "1233048625531": "alter",
    "1233048420484": "alter",
    "1233048734703": "alter",
    "1233048852562": "alter_alter",
}


def parse_questionnaire(path: Path) -> dict:
    root = ET.parse(path).getroot()
    study_id = root.get("Id")
    questions = {}
    for q in root.findall(".//QuestionList/Question"):
        qid = q.find("Id").text
        title = q.find("QuestionTitle").text
        qtype = int(q.find("QuestionType").text)
        answer_type = int(q.find("AnswerType").text)
        options = []
        for opt in q.findall(".//AnswerText"):
            options.append(
                {
                    "index": int(opt.get("index")),
                    "value": int(opt.get("value")),
                    "label": opt.text,
                    "adjacent": opt.get("adjacent") == "true",
                }
            )
        questions[qid] = {
            "id": qid,
            "variable": QUESTION_VARS.get(qid, slugify(title)),
            "title": title,
            "question_text": (q.find("QuestionText").text or "").strip(),
            "question_type": qtype,
            "question_type_name": {
                1: "ego",
                2: "alter_prompt",
                3: "alter",
                4: "alter_alter",
            }[qtype],
            "answer_type": answer_type,
            "answer_type_name": {0: "choice", 1: "numeric", 2: "text"}[answer_type],
            "options": options,
        }
    return {
        "study_id": study_id,
        "file": path.name,
        "name": root.find(".//Study/name").text,
        "num_alters": int(root.find(".//Study/numalters").text),
        "questions": questions,
    }


def slugify(title: str) -> str:
    text = re.sub(r"^\d+\.\s*", "", title).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_answer(answer: ET.Element) -> dict | None:
    answered = answer.find("Answered")
    if answered is None or answered.text != "true":
        qid = answer.find("QuestionId")
        return {
            "question_id": qid.text if qid is not None else None,
            "variable": QUESTION_VARS.get(qid.text if qid is not None else "", None),
            "answered": False,
        }

    qid = answer.find("QuestionId").text
    value_el = answer.find("Value")
    index_el = answer.find("Index")
    string_el = answer.find("String")
    adjacent_el = answer.find("Adjacent")

    result = {
        "question_id": qid,
        "variable": QUESTION_VARS.get(qid),
        "answered": True,
        "value": int(value_el.text) if value_el is not None and value_el.text is not None else None,
        "index": int(index_el.text) if index_el is not None and index_el.text is not None else None,
        "label": string_el.text if string_el is not None else None,
        "adjacent": adjacent_el.text == "true" if adjacent_el is not None else None,
    }

    alters = answer.findall(".//Alters/Index")
    if len(alters) == 1:
        result["alter_id"] = alters[0].text
    elif len(alters) == 2:
        result["source_id"] = alters[0].text
        result["target_id"] = alters[1].text

    return result


def parse_interview(path: Path, community: str) -> dict:
    root = ET.parse(path).getroot()
    study_id = root.get("StudyId")
    ego_name = root.find(".//EgoName/First").text
    alters = [node.text for node in root.findall(".//AlterList/Name")]
    answers = [parse_answer(node) for node in root.findall(".//AnswerList/Answer")]

    ego_vars: dict[str, dict | None] = {}
    alter_vars: dict[str, dict[str, dict | None]] = {alter_id: {} for alter_id in alters}
    ties: list[dict] = []
    alter_prompt = None

    for ans in answers:
        if not ans or not ans.get("variable"):
            continue
        var = ans["variable"]
        qtype = QUESTION_TYPES.get(ans["question_id"])

        if qtype == "ego":
            ego_vars[var] = ans if ans["answered"] else None
        elif qtype == "alter_prompt":
            alter_prompt = ans
        elif qtype == "alter":
            alter_id = ans.get("alter_id")
            if alter_id is not None:
                alter_vars.setdefault(alter_id, {})[var] = ans if ans["answered"] else None
        elif qtype == "alter_alter":
            ties.append(
                {
                    "source_id": ans.get("source_id"),
                    "target_id": ans.get("target_id"),
                    "value": ans.get("value"),
                    "label": ans.get("label"),
                    "adjacent": ans.get("adjacent"),
                    "probably_knows": ans.get("value") == 2,
                    "might_know": ans.get("value") == 1,
                    "probably_not": ans.get("value") == 0,
                }
            )

    return {
        "ego_id": ego_name,
        "community": community,
        "file": path.name,
        "study_id": study_id,
        "questionnaire": "pilot" if study_id == STUDY_PILOT else "main",
        "complete": (root.find("Complete").text == "true"),
        "num_alters": int(root.find("numalters").text),
        "alter_ids": alters,
        "alter_prompt": alter_prompt,
        "ego": ego_vars,
        "alters": [
            {"alter_id": alter_id, **attrs}
            for alter_id, attrs in sorted(alter_vars.items(), key=lambda item: int(item[0]))
        ],
        "alter_alter_ties": ties,
    }


def flatten_ego(record: dict) -> dict:
    row = {
        "ego_id": record["ego_id"],
        "community": record["community"],
        "questionnaire": record["questionnaire"],
        "complete": record["complete"],
        "num_alters": record["num_alters"],
    }
    for var, ans in record["ego"].items():
        if ans is None:
            row[var] = None
            row[f"{var}_label"] = None
        else:
            row[var] = ans.get("value")
            row[f"{var}_label"] = ans.get("label")
    return row


def flatten_alter(record: dict) -> list[dict]:
    rows = []
    for alter in record["alters"]:
        row = {
            "ego_id": record["ego_id"],
            "community": record["community"],
            "alter_id": alter["alter_id"],
        }
        for key, value in alter.items():
            if key == "alter_id":
                continue
            if value is None:
                row[key] = None
                row[f"{key}_label"] = None
            else:
                row[key] = value.get("value")
                row[f"{key}_label"] = value.get("label")
        rows.append(row)
    return rows


def flatten_alter_alter(record: dict) -> list[dict]:
    rows = []
    for tie in record["alter_alter_ties"]:
        rows.append(
            {
                "ego_id": record["ego_id"],
                "community": record["community"],
                "source_id": tie["source_id"],
                "target_id": tie["target_id"],
                "value": tie["value"],
                "label": tie["label"],
                "probably_knows": tie["probably_knows"],
            }
        )
    return rows


def build_value_labels(questionnaires: dict[str, dict]) -> dict:
    labels: dict[str, dict] = {}
    for questionnaire in questionnaires.values():
        for question in questionnaire["questions"].values():
            var = question["variable"]
            if not question["options"]:
                continue
            labels[var] = {
                "label": question["title"],
                "question_text": question["question_text"],
                "values": {str(opt["value"]): opt["label"] for opt in question["options"]},
            }
    return labels


def build_variable_labels(questionnaires: dict[str, dict]) -> dict:
    labels: dict[str, str] = {}
    for questionnaire in questionnaires.values():
        for question in questionnaire["questions"].values():
            labels[question["variable"]] = question["title"]
    return labels


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def export_all(processed_dir: Path = PROCESSED_DIR) -> dict:
    questionnaires = {
        "main": parse_questionnaire(SIGNOS_DIR / "signos.ego"),
        "pilot": parse_questionnaire(SIGNOS_DIR / "signos_p_piloto.ego"),
    }

    interviews: list[dict] = []
    for community in ("chinese", "filipinos", "sikhs"):
        for path in sorted((INTERVIEWS_DIR / community).glob("*.int")):
            interviews.append(parse_interview(path, community))

    ego_rows = [flatten_ego(record) for record in interviews]
    alter_rows = [row for record in interviews for row in flatten_alter(record)]
    tie_rows = [row for record in interviews for row in flatten_alter_alter(record)]

    write_json(processed_dir / "questionnaire.json", questionnaires)
    write_json(processed_dir / "variable_labels.json", build_variable_labels(questionnaires))
    write_json(processed_dir / "value_labels.json", build_value_labels(questionnaires))
    write_json(processed_dir / "ego.json", ego_rows)
    write_json(processed_dir / "alter.json", alter_rows)
    write_json(processed_dir / "alter_alter.json", tie_rows)

    for record in interviews:
        out_path = processed_dir / "by_interview" / record["community"] / f"{record['ego_id']}.json"
        write_json(out_path, record)

    manifest = {
        "source": "Signos EgoNet XML (.ego / .int)",
        "num_interviews": len(interviews),
        "by_community": {
            community: sum(1 for record in interviews if record["community"] == community)
            for community in ("chinese", "filipinos", "sikhs")
        },
        "files": {
            "questionnaire.json": "问卷定义（题目文本、选项编码）",
            "variable_labels.json": "变量名 → 西班牙语题号标题",
            "value_labels.json": "选择题变量 → 取值码说明",
            "ego.json": "70 行，每位 Ego 一行（wide）",
            "alter.json": "2100 行，每位 Ego 的每个 Alter 一行",
            "alter_alter.json": "30450 行，Alter 对之间的认识概率",
            "by_interview/{community}/{ego_id}.json": "逐人可读版本，含嵌套 ego / alters / ties",
        },
        "notes": [
            "原始 XML 仍保留在 Survey/signos/interviews/，JSON 为派生副本",
            "alter_alter 中 probably_knows=true 表示回答“Muy probablemente”（很可能认识）",
            "未回答的跳题变量在 ego.json 中为 null",
        ],
    }
    write_json(processed_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    manifest = export_all()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
