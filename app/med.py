#!/usr/bin/env python3
"""
Prescription Generator - Generate prescription PDF from JSON input

Usage:
    python3 prescription_generator.py input.json [output.pdf]
    
If output.pdf is not specified, it defaults to 'prescription.pdf'
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import random

from openai import AsyncAzureOpenAI

def generate_macro_tex(data):
    """Generate macro.tex content from JSON data"""
    patient = data.get('patient', {})
    date_info = patient.get('date', {})
    doctor = data.get('doctor', {})
    
    # Use current date if not specified
    now = datetime.now()
    year = date_info.get('year') or str(now.year)
    month = date_info.get('month') or str(now.month)
    day = date_info.get('day') or str(now.day)
    
    macro_content = f"""% User Defined Values

\\newcommand{{\\textHospitalName}}{{{data.get('hospital_name', '深圳市罗湖区人民医院')}}}
\\newcommand{{\\textPatientName}}{{{patient.get('name', '')}}}
\\newcommand{{\\textPatientGender}}{{{patient.get('gender', '女')}}}
\\newcommand{{\\textPatientAge}}{{{patient.get('age', '')}}}
\\newcommand{{\\textPatientDep}}{{{patient.get('department', '')}}}
\\newcommand{{\\textPatientID}}{{{patient.get('id', '')}}}
\\newcommand{{\\textPatientFeeType}}{{{patient.get('fee_type', '自费')}}}
\\newcommand{{\\textPatientDateYear}}{{{year}}}
\\newcommand{{\\textPatientDateMonth}}{{{month}}}
\\newcommand{{\\textPatientDateDay}}{{{day}}}
\\newcommand{{\\textPatientDiag}}{{{patient.get('diagnosis', '')}}}
\\newcommand{{\\textDoctorName}}{{{doctor.get('name', '')}}}
\\newcommand{{\\textFee}}{{{doctor.get('fee', '')}}}

% Warning: Set this value to blank may be criminal in some countries and regions.
\\newcommand{{\\textWatermark}}{{{data.get('watermark', 'test')}}}

% End

\\newcommand{{\\styleNormalText}}{{\\songti \\fontsize{{15}}{{15}} \\selectfont }}
\\newcommand{{\\blockUnderlinedText}}[1]
    {{\\uline{{\\space\\space #1 \\space\\space}}}}
\\newcommand{{\\blockRSign}}
    {{{{\\bfseries \\sffamily \\fontsize{{40}}{{40}} \\selectfont \\; R.}}}}
\\newcommand{{\\blockMedicine}}[3]{{
    {{
        \\LARGE #1
        \\hfill
        \\large #2
        \\hfill
    }}
    \\\\
    \\hspace*{{1cm}}
    {{
        \\large 用法: #3
    }}
}}"""
    return macro_content


def generate_medicine_tex(data):
    """Generate medicine.tex content from JSON data"""
    medicines = data.get('medicines', [])
    
    medicine_blocks = []
    for med in medicines:
        name = med.get('name', '')
        quantity = med.get('quantity', '')
        usage = med.get('usage', '')
        
        block = f"""\\blockMedicine{{
    {name} % 药品名称
}}
{{
    {quantity} % 药品数量
}}
{{
    {usage} % 药品用法
}}"""
        medicine_blocks.append(block)
    
    return '\n\n'.join(medicine_blocks)

# random 8 digits
random_ID = random.randint(10000000, 99999999)


def generate_main_tex():
    """Generate main.tex content - static structure"""
    main_content = r"""\documentclass[UTF8]{ctexart}
\usepackage[T1]{fontenc}
\usepackage{setspace}
\usepackage{pst-barcode}
\usepackage{tikz}
\usepackage{draftwatermark}
\usepackage{dashrule}
\usepackage[normalem]{ulem}
\usepackage[paperwidth=14.5cm,paperheight=21cm]{geometry}
\newgeometry{top=1cm,bottom=0.5cm,left=1.5cm,right=1.5cm}
\setlength\parindent{0pt}
\begin{document}
\include{macro.tex}
\DraftwatermarkOptions{text=\heiti \fbox{\textWatermark}}
\pagenumbering{gobble}

\begin{center}
    \LARGE \heiti \textHospitalName
    \\
    \Huge \heiti 处 \space 方 \space 笺
\end{center}

\vspace{0.5cm}

{
    \begin{spacing}{1.8}
    \styleNormalText
    姓 \space 名：\blockUnderlinedText{\textPatientName}
    \hfill
    性 \space 别：\blockUnderlinedText{\textPatientGender}
    \hfill
    年 \space 龄：\blockUnderlinedText{\textPatientAge}
    \hfill
    科 \space 室：\blockUnderlinedText{\textPatientDep}
    \\
    门诊号：\blockUnderlinedText{\textPatientID}
    \hfill
    费 \space 别：\blockUnderlinedText{\textPatientFeeType}
    \hfill
    日期：
    \blockUnderlinedText{\textPatientDateYear} 年
    \blockUnderlinedText{\textPatientDateMonth} 月
    \blockUnderlinedText{\textPatientDateDay} 日
    \\
    临床诊断及证型：\space \textPatientDiag 
    \end{spacing}
}

\vspace{0.5cm}
\hrule

\centerline{
\begin{minipage}{0.9\linewidth}
\vspace{0.5cm}
\blockRSign
\vspace{0.8cm}
\include{medicine.tex}
\hdashrule{\linewidth}{1pt}{5pt}
\begin{center}
    \Large (以下空白)
\end{center}
\end{minipage}
}

\vspace*{\fill}

\hrule
\vspace{0.5cm}

\styleNormalText
医 \space 师：\blockUnderlinedText{\qquad\textDoctorName\qquad}
\hfill
金 \space 额：
\blockUnderlinedText{\qquad\textFee\qquad}
\\
药师（审核、校对、发药）：\blockUnderlinedText{\qquad 张倩 \qquad}
\hfill
药师/士（调配）：\blockUnderlinedText{\qquad\qquad}

\vspace{0.5cm}

\begin{minipage}{0.7\linewidth}
\begin{spacing}{1}
    温馨提示：
    \begin{enumerate}
        \itemsep0em 
        \item 本处方当天有效，过期作废；
        \item 取药时请仔细核对清单，点齐药品；
        \item 依《电报药品管理法》，药品一经发出，一律不得退换；
    \end{enumerate}
    支付宝，微信（仅限自费和普通医保）
\end{spacing}
\end{minipage}

\begin{tikzpicture}[remember picture,overlay]
    \node[xshift=-4cm,yshift=-2cm] at (current page.north east){
        \LARGE \heiti \fbox {普通}
    };
    \node[xshift=2cm,yshift=-3cm] at (current page.north west){
        \psbarcode{*11451419*}{includetext width=2 height=0.5 textsize=15 textgaps=2}{code128}
    };
    \node[xshift=-4cm,yshift=1cm] at (current page.south east){
        \psbarcode{ScanThisToPayYourFeeOfMedicinesAndEnjoyYourLife}{includetext width=1 height=1}{qrcode}
    };
    \node[xshift=6.5cm,yshift=6cm] at (current page.south west){
        \includegraphics[width=4cm,angle=6]{data/sign.png}
    };
\end{tikzpicture}

\pagenumbering{gobble}
\end{document}"""
    # replace the placeholder numeric ID in the LaTeX with the generated random_ID
    main_content = main_content.replace("11451419", str(random_ID))
    return main_content


async def _write_text_async(path: Path, content: str) -> None:
    await asyncio.to_thread(path.write_text, content, encoding='utf-8')


async def _copy_async(src: Path, dst: Path) -> None:
    await asyncio.to_thread(shutil.copy, src, dst)


async def _copytree_async(src: Path, dst: Path) -> None:
    await asyncio.to_thread(shutil.copytree, src, dst)


async def _ensure_dir_async(path: Path) -> None:
    await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True)


async def generate_pdf(json_input, output_pdf=None):
    """
    Generate a prescription PDF from JSON input
    
    Args:
        json_file: Path to input JSON file
        output_pdf: Path to output PDF file (default: prescription.pdf)
    
    Returns:
        True if successful, False otherwise
    """
    data = json_input

    if output_pdf is None:
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = f'output/prescription_{time}.pdf'

    output_path = Path(output_pdf)
    script_dir = Path(__file__).parent.absolute()
    images_dir = script_dir / 'data'

    if not images_dir.exists():
        print(f"Warning: images directory not found at {images_dir}")
        print("The PDF generation may fail if images are required.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        macro_content = generate_macro_tex(data)
        medicine_content = generate_medicine_tex(data)
        main_content = generate_main_tex()

        await asyncio.gather(
            _write_text_async(tmpdir_path / 'macro.tex', macro_content),
            _write_text_async(tmpdir_path / 'medicine.tex', medicine_content),
            _write_text_async(tmpdir_path / 'main.tex', main_content)
        )

        if images_dir.exists():
            await _copytree_async(images_dir, tmpdir_path / 'data')

        try:
            last_stdout = ''
            last_stderr = ''
            for i in range(2):
                try:
                    process = await asyncio.create_subprocess_exec(
                        'xelatex', '-interaction=nonstopmode', 'main.tex',
                        cwd=str(tmpdir_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                except FileNotFoundError:
                    print("Error: xelatex not found. Please install TeX Live or similar LaTeX distribution.")
                    print("On Ubuntu/Debian: sudo apt-get install texlive-xetex texlive-latex-extra")
                    return False

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=60)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.communicate()
                    print("Error: LaTeX compilation timed out")
                    return False

                last_stdout = stdout_bytes.decode('utf-8', 'ignore')
                last_stderr = stderr_bytes.decode('utf-8', 'ignore')

                if process.returncode != 0 and i == 1:
                    print("Error compiling LaTeX:")
                    if last_stdout:
                        print(last_stdout)
                    if last_stderr:
                        print(last_stderr)
                    return False

            src_pdf = tmpdir_path / 'main.pdf'
            if src_pdf.exists():
                await _ensure_dir_async(output_path.parent)
                await _copy_async(src_pdf, output_path)
                print(f"PDF generated successfully: {output_path}")
                return output_path

            print("Error: PDF file was not generated")
            if last_stdout:
                print(last_stdout)
            if last_stderr:
                print(last_stderr)
            return False

        except Exception as e:
            print(f"Error during compilation: {e}")
            return False

sample_input = {
    "hospital_name": "深圳市罗湖区人民医院",
    "patient": {
        "name": "王小美",
        "gender": "女",
        "age": "22岁",
        "department": "精神科综合门诊",
        "id": "000114514",
        "fee_type": "自费",
        "date": {
            "year": "2025",
            "month": "10",
            "day": "11"
        },
        "diagnosis": "焦虑状态, 抑郁状态"
    },
    "medicines": [
        {
            "name": "盐酸氟西汀胶囊 20mg",
            "quantity": "1 盒",
            "usage": "\\quad 20mg \\quad 口服 \\quad 每日一次 \\quad 14天"
        },
    ],
    "doctor": {
        "name": "孙致连",
        "fee": "30.00 元"
    },
    "watermark": ""
}

async def generate_med(
    prompt: str,
    AZURE_OPENAI_ENDPOINT: str,
    AZURE_OPENAI_API_KEY: str,
    AZURE_OPENAI_API_VERSION: str,
    AZURE_OPENAI_DEPLOYMENT_NAME: str,
) -> dict:
    """Use Azure OpenAI to turn a natural-language prescription brief into JSON for ``generate_pdf``."""

    schema_template = """
Return a JSON object that strictly follows this schema:
{
    "hospital_name": "string",
    "patient": {
        "name": "string",
        "gender": "string",
        "age": "string",
        "department": "string",
        "id": "string",
        "fee_type": "string",
        "date": {
            "year": "string",
            "month": "string",
            "day": "string"
        },
        "diagnosis": "string"
    },
    "medicines": [
        {
            "name": "string",
            "quantity": "string",
            "usage": "string"
        }
    ],
    "doctor": {
        "name": "string",
        "fee": "string"
    },
    "watermark": "string"
}

Rules:
- Always include every field shown above.
- Use empty strings when the prompt does not supply a value.
- Output must be valid JSON with double-quoted keys and string values.
- Represent numbers as strings (e.g., "30.00 元").
- The ``medicines`` array must contain at least one entry; synthesize reasonable defaults if necessary.
- If the prompt omits department, use "精神科综合门诊".
- If the prompt omits patient ID, generate a random 10-digit string.
- If the prompt omits fee type, use "自费".
- If the prompt omits the date, default to year "2025", month "10", and day "11".
- Keep the doctor name exactly "孙致连" unless explicitly overridden.
- Use an empty string for the watermark unless the prompt provides a value.
- Do not add extra fields or commentary.
- If not enough information is provided, synthesize reasonable defaults to complete the JSON.
- If hospital name is missing, use "北京大学第三医院".

Sample real JSON:
"""



    schema_instructions = f"{schema_template}{str(sample_input)}\nUser prompt: {prompt}"


    client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    response = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a meticulous medical scribe who outputs structured prescription JSON.",
            },
            {"role": "user", "content": schema_instructions},
        ],
    )

    content = response.choices[0].message.content.strip()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {content}") from exc

    required_top_level = {"hospital_name", "patient", "medicines", "doctor", "watermark"}
    missing = required_top_level.difference(payload.keys())
    if missing:
        raise ValueError(f"Model response missing fields: {sorted(missing)}")

    patient = payload.get("patient")
    if not isinstance(patient, dict):
        raise ValueError("Model response must include a patient object")

    doctor = payload.get("doctor")
    if not isinstance(doctor, dict):
        raise ValueError("Model response must include a doctor object")

    medicines = payload.get("medicines")
    if not isinstance(medicines, list) or not medicines:
        raise ValueError("Model response must include at least one medicine entry")

    if not all(isinstance(item, dict) for item in medicines):
        raise ValueError("Each medicine entry must be a JSON object")

    patient.setdefault("department", "精神科综合门诊")

    patient_id = patient.get("id")
    if not patient_id:
        patient["id"] = "".join(random.choices("0123456789", k=10))

    patient.setdefault("fee_type", "自费")

    default_date = {"year": "2025", "month": "10", "day": "11"}
    date_info = patient.get("date")
    if not isinstance(date_info, dict):
        patient["date"] = default_date.copy()
    else:
        for key, value in default_date.items():
            if not date_info.get(key):
                date_info[key] = value

    doctor.setdefault("name", "孙致连")
    doctor.setdefault("fee", "")

    payload.setdefault("watermark", "")

    for med in medicines:
        med.setdefault("name", "")
        med.setdefault("quantity", "")
        med.setdefault("usage", "")

    return payload




async def generate_jpg(pdf_path, jpg_output=None, *, quality=30, ppi=150):
    """Generate a JPG from the first page of the given PDF using ImageMagick."""
    src_path = Path(pdf_path)
    if not src_path.exists():
        print(f"Error: source PDF not found at {src_path}")
        return False

    if jpg_output is None:
        jpg_output = src_path.with_suffix('.jpg')

    if quality <= 0 or quality > 100:
        print("Error: quality must be within 1-100")
        return False

    if ppi <= 0:
        print("Error: ppi must be positive")
        return False

    output_path = Path(jpg_output)
    await _ensure_dir_async(output_path.parent)

    try:
        process = await asyncio.create_subprocess_exec(
            'magick',
            '-density', str(ppi),
            str(src_path) + '[0]',
            '-quality', str(quality),
            '-units', 'PixelsPerInch',
            '-flatten',
            str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        print("Error: ImageMagick 'magick' command not found. Please install ImageMagick to enable JPG generation.")
        return False

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=60)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        print("Error: JPG conversion timed out")
        return False

    stdout_text = stdout_bytes.decode('utf-8', 'ignore')
    stderr_text = stderr_bytes.decode('utf-8', 'ignore')

    if process.returncode != 0:
        print("Error converting PDF to JPG:")
        if stdout_text:
            print(stdout_text)
        if stderr_text:
            print(stderr_text)
        return False

    if stdout_text:
        print(stdout_text)
    if stderr_text:
        print(stderr_text)

    print(f"JPG generated successfully: {output_path}")

    # Delete the PDF after successful JPG generation when paths differ
    if src_path != output_path:
        try:
            src_path.unlink()
            print(f"Deleted temporary PDF file: {src_path}")
        except Exception as e:
            print(f"Warning: could not delete temporary PDF file: {e}")

    return output_path

# Legacy alias for compatibility
generate_jpg_med = generate_jpg


# directly create jpg from JSON by calling generate_pdf and generate_jpg
async def generate_jpg_from_med_json(
        json_input,
        output_jpg,
    ):
    generate_pdf_path = await generate_pdf(json_input, None)
    print(f"Generated PDF path: {generate_pdf_path}")
    if not generate_pdf_path:
        return False
    jpg_path = await generate_jpg(generate_pdf_path, output_jpg)
    return jpg_path

    


async def main():
    """Main entry point"""
    output_pdf = None  # or specify a path like 'output.pdf'

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    data = sample_input

    if prompt:
        env_map = {
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        }
        missing = [name for name, value in env_map.items() if not value]
        if missing:
            print(f"Error: missing Azure OpenAI configuration environment variables: {', '.join(missing)}")
            return 1

        try:
            data = await generate_med(
                prompt,
                env_map["AZURE_OPENAI_ENDPOINT"],
                env_map["AZURE_OPENAI_API_KEY"],
                env_map["AZURE_OPENAI_API_VERSION"],
                env_map["AZURE_OPENAI_DEPLOYMENT_NAME"],
            )
        except Exception as exc:
            print(f"Error generating prescription JSON: {exc}")
            return 1

    pdf_path = await generate_pdf(data, output_pdf)
    if not pdf_path:
        return 1

    success = await generate_jpg(pdf_path)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
