from app.med import generate_macro_tex, generate_main_tex, generate_medicine_tex


def test_generate_macro_tex_includes_patient_data():
    data = {
        "hospital_name": "H",
        "patient": {
            "name": "Alice",
            "gender": "女",
            "age": "20",
            "department": "D",
            "id": "123",
            "fee_type": "自费",
            "date": {"year": "2026", "month": "3", "day": "26"},
            "diagnosis": "Diag",
            "catagory": "普通",
        },
        "doctor": {"name": "Doc", "fee": "10.00 元"},
        "watermark": "wm",
    }
    tex = generate_macro_tex(data)
    assert "\\newcommand{\\textHospitalName}{H}" in tex
    assert "\\newcommand{\\textPatientName}{Alice}" in tex
    assert "\\newcommand{\\textDoctorName}{Doc}" in tex


def test_generate_medicine_tex_contains_all_entries():
    data = {
        "medicines": [
            {"name": "A", "quantity": "1", "usage": "u1", "price": "2"},
            {"name": "B", "quantity": "3", "usage": "u2", "price": "4"},
        ]
    }
    tex = generate_medicine_tex(data)
    assert "A % 药品名称" in tex
    assert "B % 药品名称" in tex
    assert tex.count("\\blockMedicine") == 2


def test_generate_macro_tex_escapes_latex_special_characters():
    data = {
        "hospital_name": "H & 50%",
        "patient": {
            "name": "A_B #1",
            "gender": "女",
            "age": "20",
            "department": "D{1}",
            "id": "123_45",
            "fee_type": "自费",
            "date": {"year": "2026", "month": "4", "day": "30"},
            "diagnosis": "焦虑 & 抑郁 50%",
            "catagory": "普通",
        },
        "doctor": {"name": "Doc_1", "fee": "10.00 元"},
        "watermark": "wm#1",
    }
    tex = generate_macro_tex(data)

    assert "H \\& 50\\%" in tex
    assert "A\\_B \\#1" in tex
    assert "D\\{1\\}" in tex
    assert "\\newcommand{\\textPatientID}{12345}" in tex
    assert "焦虑 \\& 抑郁 50\\%" in tex
    assert "Doc\\_1" in tex
    assert "wm\\#1" in tex


def test_generate_medicine_tex_escapes_latex_special_characters():
    data = {
        "medicines": [
            {
                "name": "A_B & 50%",
                "quantity": "2#",
                "usage": "after_meal {daily}",
                "price": "80.50 元",
            }
        ]
    }
    tex = generate_medicine_tex(data)

    assert "A\\_B \\& 50\\%" in tex
    assert "2\\#" in tex
    assert "after\\_meal \\{daily\\}" in tex


def test_generate_main_tex_uses_xecjk_with_bundled_fonts():
    tex = generate_main_tex()

    assert "\\documentclass{article}" in tex
    assert "\\usepackage{xeCJK}" in tex
    assert "FandolSong-Regular.otf" in tex
    assert "FandolHei-Regular.otf" in tex
    assert "ctexart" not in tex
