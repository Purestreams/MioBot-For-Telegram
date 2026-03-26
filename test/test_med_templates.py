from app.med import generate_macro_tex, generate_medicine_tex


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
