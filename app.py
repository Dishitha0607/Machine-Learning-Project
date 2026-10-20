"""
Root entry point for Streamlit application.
Loads and executes the main application from app/main.py.
"""
import os
import sys

# Ensure root and app directories are in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root_dir, "app")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Execute app/main.py directly
main_script_path = os.path.join(app_dir, "main.py")
with open(main_script_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), main_script_path, "exec")
    exec(code, globals())
