# C:\Users\anapa\SuperIA\EzioFilhoUnified\organize_models.py
"""
Auto-organiza todos os modelos e gera relatório detalhado.
"""

import os
import shutil

base_dir = r"C:\Users\anapa\SuperIA\EzioFilhoUnified\modelos_hf"
inventory_file = r"C:\Users\anapa\SuperIA\EzioFilhoUnified\model_inventory.txt"

def get_model_folder(file_name):
    # Melhora: se já estiver em subpasta, mantém, senão usa prefixo do arquivo
    if "--" in file_name:
        return file_name.split("--")[1].split("-")[0]
    if "-" in file_name:
        return file_name.split("-")[0]
    if "_" in file_name:
        return file_name.split("_")[0]
    return file_name.split(".")[0]

def move_file_to_model_folder(file_path, actions):
    name = os.path.basename(file_path)
    parent = os.path.dirname(file_path)
    if os.path.dirname(parent).endswith("modelos_hf"):
        actions.append(f"[SKIP] Already organized: {file_path}")
        return
    folder = os.path.basename(parent)
    if folder in ["modelos_hf"]:
        folder = get_model_folder(name)
    dest_folder = os.path.join(base_dir, folder)
    os.makedirs(dest_folder, exist_ok=True)
    dest_file = os.path.join(dest_folder, name)
    if file_path != dest_file:
        shutil.move(file_path, dest_file)
        actions.append(f"[MOVE] {file_path} -> {dest_file}")
    else:
        actions.append(f"[OK] {file_path}")

def scan_and_move():
    actions = []
    model_summary = {}
    extensions = (".gguf", ".bin", ".safetensors")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                move_file_to_model_folder(file_path, actions)
                folder = os.path.basename(os.path.dirname(file_path))
                model_summary.setdefault(folder, []).append(file)
    # Relatório
    with open(inventory_file, "w", encoding="utf-8") as f:
        f.write("Model Inventory Report\n\n")
        for folder, files in sorted(model_summary.items()):
            f.write(f"## {folder}\n")
            for file in files:
                f.write(f" - {file}\n")
            f.write("\n")
        f.write("ACTIONS LOG:\n")
        for act in actions:
            f.write(act + "\n")
    print(f"Finished organizing models. See inventory: {inventory_file}")

if __name__ == "__main__":
    scan_and_move()
