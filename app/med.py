#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a prescription PDF from a JSON file.

Usage:
  python prescription_template.py input.json output.pdf

Notes:
- For Chinese characters, provide a CJK font file (e.g., NotoSansCJKsc-Regular.otf or SimSun.ttf)
  via --font path/to/font.ttf or the JSON field header.font_path.
- This template is for demonstration only and not a real medical document.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4, A5
from reportlab.lib.units import mm
from reportlab.lib import colors


def register_font(font_path: Optional[str]) -> str:
    """
    Register a font for multilingual text. Returns the font name to use.
    """
    if font_path and os.path.exists(font_path):
        font_name = "CJKMain"
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            return font_name
        except Exception as e:
            print(f"[warn] Failed to register font at {font_path}: {e}. Falling back to Helvetica.")
    # Fallback; may not render CJK properly
    return "Helvetica"


def ensure_page_size(name: Optional[str]):
    """
    Return a reportlab page size by common name.
    """
    if not name:
        return A5  # closer to the example size
    name = str(name).strip().upper()
    if name == "A4":
        return A4
    if name == "A5":
        return A5
    # default
    return A5


def draw_checkbox(c: canvas.Canvas, x: float, y: float, label: str, checked: bool, font_name: str, font_size: int = 9):
    box_size = 4.2 * mm
    c.setLineWidth(0.7)
    c.rect(x, y - box_size, box_size, box_size)
    if checked:
        c.setStrokeColor(colors.black)
        c.line(x + 0.8 * mm, y - box_size + 0.8 * mm, x + box_size - 0.8 * mm, y - 0.8 * mm)
        c.line(x + box_size - 0.8 * mm, y - box_size + 0.8 * mm, x + 0.8 * mm, y - 0.8 * mm)
    c.setFont(font_name, font_size)
    c.drawString(x + box_size + 1.2 * mm, y - box_size + 0.6 * mm, label)


def text_wrap(c: canvas.Canvas, text: str, max_width: float, font_name: str, font_size: int) -> List[str]:
    """
    Simple word-based wrapping for Latin text; for CJK, splits by character.
    """
    if not text:
        return []
    c.setFont(font_name, font_size)
    # Heuristic: if contains CJK, split by char
    has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text)
    if has_cjk:
        lines, cur, cur_w = [], "", 0
        for ch in text:
            w = pdfmetrics.stringWidth(ch, font_name, font_size)
            if cur_w + w <= max_width:
                cur += ch
                cur_w += w
            else:
                if cur:
                    lines.append(cur)
                cur, cur_w = ch, w
        if cur:
            lines.append(cur)
        return lines
    else:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            test = (cur + " " + w).strip()
            if pdfmetrics.stringWidth(test, font_name, font_size) <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines


def draw_header(c: canvas.Canvas, data: Dict[str, Any], font_name: str, page_w: float, top_y: float) -> float:
    header = data.get("header", {})
    left_logo_path = header.get("logo_left_path")
    right_logo_path = header.get("logo_right_path")
    title = header.get("title", "处方笺")
    hospital = header.get("hospital", "")
    affiliation = header.get("affiliation", "")
    number = header.get("no", "")

    # optional logos
    logo_h = 12 * mm
    margin = 15 * mm
    y = top_y

    if left_logo_path and os.path.exists(left_logo_path):
        try:
            c.drawImage(left_logo_path, margin, y - logo_h, height=logo_h, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            print(f"[warn] Failed to draw left logo: {e}")

    if right_logo_path and os.path.exists(right_logo_path):
        try:
            c.drawImage(right_logo_path, page_w - margin - 20 * mm, y - logo_h, height=logo_h, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            print(f"[warn] Failed to draw right logo: {e}")

    # Hospital and affiliation line
    if affiliation:
        c.setFont(font_name, 10)
        c.drawCentredString(page_w / 2.0, y - 3 * mm, affiliation)
        y -= 10 * mm
    if hospital:
        c.setFont(font_name, 18)
        c.drawCentredString(page_w / 2.0, y - 3 * mm, hospital+title)
        y -= 7 * mm

    # Title
    #c.setFont(font_name, 18)
    #c.drawCentredString(page_w / 2.0, y - 2 * mm, title)
    #y -= 14 * mm

    # Number
    if number:
        c.setFont(font_name, 10)
        c.drawRightString(page_w - margin, y, f"No {number}")

    # horizontal rule
    c.setLineWidth(0.6)
    c.line(margin, y - 2 * mm, page_w - margin, y - 2 * mm)
    return y - 10 * mm


def draw_patient_section(c: canvas.Canvas, data: Dict[str, Any], font_name: str, page_w: float, cur_y: float) -> float:
    margin = 15 * mm
    line_h = 7 * mm

    fee_type = data.get("fee_type", "")
    fee_opts = ["公费", "自费", "医保", "其他"]
    x = margin
    c.setFont(font_name, 10)
    c.drawString(x, cur_y, "费别:")
    x += 12 * mm
    for opt in fee_opts:
        draw_checkbox(c, x, cur_y + 4.2 * mm, opt, fee_type == opt, font_name, 9)
        x += 18 * mm
    cur_y -= line_h

    patient = data.get("patient", {})
    name = patient.get("name", "")
    gender = patient.get("gender", "")
    age = patient.get("age", "")
    visit_no = patient.get("visit_no", "")
    dept = patient.get("dept", "")
    diagnosis = patient.get("diagnosis", "")
    id_no = patient.get("id_no", "")
    phone = patient.get("phone", "")

    # Row: Name, Gender, Age
    c.drawString(margin, cur_y, f"姓名: {name}")
    c.drawString(margin + 50 * mm, cur_y, f"性别: {gender}")
    c.drawString(margin + 85 * mm, cur_y, f"年龄: {age}")
    cur_y -= line_h

    # Row: 门诊/住院病历号, 科别(病区/床位号)
    c.drawString(margin, cur_y, f"门诊/住院病历号: {visit_no}")
    c.drawString(margin + 80 * mm, cur_y, f"科别(病区/床位号): {dept}")
    cur_y -= line_h

    # Row: 临床诊断, 开具日期
    today = data.get("date") or datetime.now().strftime("%Y-%m-%d")
    c.drawString(margin, cur_y, f"临床诊断: {diagnosis}")
    c.drawString(margin + 80 * mm, cur_y, f"开具日期: {today}")
    cur_y -= line_h

    # Row: 地址/电话 & 身份证/医保号
    c.drawString(margin, cur_y, f"身份证/医保号: {id_no}")
    c.drawString(margin + 80 * mm, cur_y, f"电话: {phone}")
    cur_y -= line_h

    # Rp label
    c.setFont(font_name, 12)
    c.drawString(margin, cur_y - 1 * mm, "Rp")
    # divider line under header portion
    c.setLineWidth(0.4)
    c.line(margin, cur_y - 3 * mm, page_w - margin, cur_y - 3 * mm)
    return cur_y - 7 * mm


def format_item_line(item: Dict[str, Any]) -> str:
    """
    Compose a standard line for one prescription item.
    """
    parts = []
    if item.get("name"):
        parts.append(str(item["name"]))
    if item.get("strength"):
        parts.append(str(item["strength"]))
    if item.get("form"):
        parts.append(str(item["form"]))
    if item.get("dose"):
        parts.append(f"每次 {item['dose']}")
    if item.get("route"):
        parts.append(str(item["route"]))
    if item.get("frequency"):
        parts.append(str(item["frequency"]))
    if item.get("days"):
        parts.append(f"{item['days']} 天")
    if item.get("quantity"):
        parts.append(f"数量 {item['quantity']}")
    if item.get("notes"):
        parts.append(f"({item['notes']})")
    return "  ".join(parts)


def draw_items(c: canvas.Canvas, items: List[Dict[str, Any]], font_name: str, page_w: float, cur_y: float) -> float:
    margin = 22 * mm  # indent to align after "Rp"
    right_margin = 15 * mm
    max_w = page_w - margin - right_margin
    c.setFont(font_name, 11)

    if not items:
        c.drawString(margin, cur_y, "—")
        return cur_y - 8 * mm

    idx = 1
    for it in items:
        line = it.get("line") or format_item_line(it)
        # wrap
        lines = text_wrap(c, line, max_w, font_name, 11)
        if not lines:
            lines = ["—"]
        prefix = f"{idx}."
        first_line = f"{prefix} {lines[0]}"
        c.drawString(margin, cur_y, first_line)
        cur_y -= 6.5 * mm
        for cont in lines[1:]:
            c.drawString(margin + pdfmetrics.stringWidth(prefix + " ", font_name, 11), cur_y, cont)
            cur_y -= 6.5 * mm
        idx += 1
        cur_y -= 2 * mm
    return cur_y


def draw_footer(c: canvas.Canvas, data: Dict[str, Any], font_name: str, page_w: float, cur_y: float):
    margin = 15 * mm
    c.setLineWidth(0.4)
    c.line(margin, cur_y, page_w - margin, cur_y)
    cur_y -= 7 * mm

    doctor = data.get("doctor", {})
    pharmacist = data.get("pharmacist", {})
    amount = data.get("amount", "")

    c.setFont(font_name, 10)
    c.drawString(margin, cur_y, f"医师：{doctor.get('name', '')}")
    c.drawString(margin + 50 * mm, cur_y, f"药品金额：{amount}")
    cur_y -= 7 * mm

    c.drawString(margin, cur_y, f"审核药师：{pharmacist.get('checker', '')}")
    c.drawString(margin + 50 * mm, cur_y, f"调配药师/士：{pharmacist.get('dispenser', '')}")
    cur_y -= 7 * mm

    c.drawString(margin, cur_y, f"核对、发药药师：{pharmacist.get('verifier', '')}")
    cur_y -= 10 * mm

    c.setFont(font_name, 9)
    c.drawCentredString(page_w / 2.0, cur_y, "处方为开具当日有效")
    cur_y -= 6 * mm

    printed_on = data.get("printed_on") or datetime.now().strftime("%Y年%m月%d日")
    c.drawRightString(page_w - margin, cur_y, f"印制日期：{printed_on}")


def generate_pdf(data: Dict[str, Any], out_path: str, font_path: Optional[str] = None):
    page_size = ensure_page_size(data.get("page_size"))
    page_w, page_h = page_size
    c = canvas.Canvas(out_path, pagesize=page_size)

    # Register font and set default font
    if not font_path:
        font_path = (data.get("header", {}) or {}).get("font_path")
    font_name = register_font(font_path)

    top_y = page_h - 15 * mm
    y = draw_header(c, data, font_name, page_w, top_y)
    y = draw_patient_section(c, data, font_name, page_w, y)

    items = data.get("items", [])
    y = draw_items(c, items, font_name, page_w, y)

    draw_footer(c, data, font_name, page_w, y)
    c.showPage()
    c.save()
    print(f"[ok] Wrote {out_path}")


sample_data = {
  "page_size": "A5",
  "header": {
    "affiliation": "广东省医学科学院",
    "hospital": "广东省人民医院",
    "title": "处方笺",
    "no": "3549524",
    "font_path": "NotoSansCJKsc-Regular.otf",
    "logo_left_path": "",
    "logo_right_path": ""
  },
  "fee_type": "医保",
  "date": "2024-08-11",
  "patient": {
    "name": "张三",
    "gender": "男",
    "age": "22 岁",
    "visit_no": "MZ20240811001",
    "dept": "内科/12床",
    "diagnosis": "上呼吸道感染",
    "id_no": "350781200001010011",
    "phone": "13450000000"
  },
  "items": [
    {
      "name": "阿莫西林",
      "strength": "500 mg",
      "form": "胶囊",
      "dose": "1 粒",
      "route": "口服",
      "frequency": "每日 3 次",
      "days": 7,
      "quantity": 21,
      "notes": "饭后服用"
    },
    {
      "name": "布洛芬",
      "strength": "200 mg",
      "form": "片剂",
      "dose": "1 片",
      "route": "口服",
      "frequency": "必要时",
      "days": 3,
      "quantity": 6,
      "notes": "发热或疼痛时服用"
    }
  ],
  "amount": "¥58.00",
  "doctor": {
    "name": "李医生"
  },
  "pharmacist": {
    "checker": "王药师",
    "dispenser": "赵药师",
    "verifier": "孙药师"
  },
  "printed_on": "2025年08月11日"
}


def main():
    #parser = argparse.ArgumentParser(description="Generate a prescription PDF from JSON.")
    #parser.add_argument("input", help="Path to JSON data")
    #parser.add_argument("output", help="Path to output PDF")
    #parser.add_argument("--font", help="Path to a TTF/OTF font (e.g., NotoSansCJKsc-Regular.otf)", default=None)
    #args = parser.parse_args()

    #with open(args.input, "r", encoding="utf-8") as f:
    #    data = json.load(f)

    generate_pdf(sample_data, "output/prescription_sample.pdf", font_path="data/NotoSansSC-Regular.ttf")


if __name__ == "__main__":
    main()