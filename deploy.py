"""Un solo comando:  python deploy.py

1. converte cards.pt (YOLO) in ONNX
2. cifra il modello con la TUA password (cards.enc): senza password l'app non funziona
3. pubblica la web app su GitHub Pages
Requisiti: Python 3, git e GitHub CLI (gh) installati.
"""
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "poker-equity"
os.chdir(Path(__file__).parent)


def run(*args, check=True, capture=False, env=None):
    r = subprocess.run(args, check=check, capture_output=capture, text=True, env=env)
    return r.stdout.strip() if capture else r.returncode


def venv_python():
    if not Path(".venv").exists():
        run(sys.executable, "-m", "venv", ".venv")
    return str(Path(".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"))


# 1) modello -> ONNX
if Path("cards.pt").exists() and not Path("cards.onnx").exists():
    py = venv_python()
    run(py, "-m", "pip", "install", "-q", "ultralytics", "onnx")
    export = (
        "import json, shutil\n"
        "from ultralytics import YOLO\n"
        "m = YOLO('cards.pt')\n"
        "p = m.export(format='onnx', imgsz=640, simplify=True, opset=12)\n"
        "shutil.move(p, 'cards.onnx')\n"
        "json.dump([m.names[i] for i in sorted(m.names)], open('labels.json', 'w'))\n"
    )
    run(py, "-c", export)

if not (Path("cards.onnx").exists() and Path("labels.json").exists()):
    sys.exit("Manca cards.pt (modello YOLO delle carte) in questa cartella.")

# 2) cifratura con password (AES-256-GCM, chiave da PBKDF2-SHA256 600k iterazioni)
password = os.environ.get("POKER_PW")
if not password:
    print("La password e' VISIBILE mentre la scrivi (cosi' eviti errori di battitura). Annotala.")
    while True:
        password = input("Scegli la password dell'app (min. 10 caratteri), poi premi Invio: ").strip()
        if len(password) < 10:
            print("Troppo corta: una password debole si indovina anche se il modello e' cifrato.")
            continue
        break

py = venv_python()
run(py, "-m", "pip", "install", "-q", "cryptography")
encrypt = (
    "import os, secrets\n"
    "from cryptography.hazmat.primitives import hashes\n"
    "from cryptography.hazmat.primitives.ciphers.aead import AESGCM\n"
    "from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC\n"
    "salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)\n"
    "key = PBKDF2HMAC(hashes.SHA256(), 32, salt, 600000).derive(os.environ['POKER_PW'].encode())\n"
    "data = AESGCM(key).encrypt(iv, open('cards.onnx', 'rb').read(), None)\n"
    "open('cards.enc', 'wb').write(salt + iv + data)\n"
)
run(py, "-c", encrypt, env={**os.environ, "POKER_PW": password})
print("Modello cifrato -> cards.enc (la password non e' recuperabile: se la perdi, rilancia deploy.py)")

# 3) pubblicazione
for tool, hint in (("git", "https://git-scm.com/downloads"), ("gh", "https://cli.github.com")):
    if not shutil.which(tool):
        sys.exit(f"Installa {tool}: {hint}")

if subprocess.run(["gh", "auth", "status"], capture_output=True).returncode != 0:
    run("gh", "auth", "login")  # una volta sola

user = run("gh", "api", "user", "-q", ".login", capture=True)
git = ["git", "-c", f"user.name={user}", "-c", f"user.email={user}@users.noreply.github.com"]

if not Path(".git").exists():
    run("git", "init", "-q")
    run("git", "branch", "-M", "main")
run("git", "add", "-A")
subprocess.run(git + ["commit", "-q", "-m", "Poker Equity web app"], check=False)

has_remote = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True).returncode == 0
if has_remote:
    run("git", "push", "-q", "origin", "main")
else:
    run("gh", "repo", "create", REPO, "--public", "--source", ".", "--remote", "origin", "--push")

# abilita GitHub Pages (se esiste gia', l'errore e' innocuo)
subprocess.run(["gh", "api", "-X", "POST", f"repos/{user}/{REPO}/pages",
                "-f", "source[branch]=main", "-f", "source[path]=/"], capture_output=True)

print(f"\nFatto! Tra 1-2 minuti apri da iPhone (Safari):\nhttps://{user}.github.io/{REPO}/")
print("Poi: Condividi > Aggiungi alla schermata Home.")
