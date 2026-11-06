# -*- coding: utf-8 -*-
"""
Hoan tat bao cao sau khi co ket qua tu Kaggle.

    python tools/finalize_report.py <thu muc chua ket_qua.zip>

Lam: giai nen ket qua vao du an, sinh lai cac bang so lieu, bien dich ca hai
ban bao cao (tieng Viet va tieng Anh) va don dep file trung gian.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")
TECTONIC = os.environ.get("TECTONIC",
                          r"C:\Users\b\AppData\Local\Temp\opencode\tectonic_msvc\tectonic.exe")


def extract(zip_path: str) -> None:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ROOT)
    print("Da giai nen:", zip_path)


def gen_tables() -> None:
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_tables.py")],
                   check=True, cwd=ROOT)


def compile_pdf(name: str) -> None:
    if not os.path.exists(TECTONIC):
        print("Khong co tectonic, bo qua bien dich", name)
        return
    r = subprocess.run([TECTONIC, "-X", "compile", name, "--outdir", "."],
                       cwd=REPORT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if "error" in line.lower() or "Overfull" in line or "Writing" in line:
            print("   ", line.strip())
    pdf = os.path.join(REPORT, name.replace(".tex", ".pdf"))
    if os.path.exists(pdf):
        print(f"    -> {name.replace('.tex', '.pdf')}: {os.path.getsize(pdf):,} bytes")
    else:
        print(f"    !! khong tao duoc {pdf}")


def cleanup() -> None:
    for fn in ["BAO_CAO.log", "BAO_CAO.aux", "BAO_CAO.xdv",
               "BAO_CAO_EN.log", "BAO_CAO_EN.aux", "BAO_CAO_EN.xdv"]:
        p = os.path.join(REPORT, fn)
        if os.path.exists(p):
            os.remove(p)
    for root, dirs, _files in os.walk(ROOT):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
        if ".pytest_cache" in root:
            shutil.rmtree(root, ignore_errors=True)
    print("Da don dep file trung gian")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Dung: python tools/finalize_report.py <ket_qua.zip>")
    extract(sys.argv[1])
    gen_tables()
    print("Bien dich bao cao tieng Viet:")
    compile_pdf("BAO_CAO.tex")
    print("Bien dich bao cao tieng Anh:")
    compile_pdf("BAO_CAO_EN.tex")
    cleanup()

    summary = os.path.join(ROOT, "saved_models", "bang_tom_tat.txt")
    if os.path.exists(summary):
        print("\n" + open(summary, encoding="utf-8").read())


if __name__ == "__main__":
    main()
