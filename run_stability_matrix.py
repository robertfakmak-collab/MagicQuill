import os
import sys

path = r"C:\Users\rober\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Stability Matrix.lnk"

if sys.platform == "win32":
    os.startfile(path)
else:
    print(f"To polecenie zadziała tylko w systemie Windows. Próba otwarcia pliku: {path}")
