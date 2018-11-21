import subprocess
import sys

my_path = "C:\Users\Rich\Downloads\scikit_cuda-0.5.2-py2.py3-none-any.whl"
command_list = [sys.executable, "-m", "pip", "install", my_path]
with subprocess.Popen(command_list, stdout=subprocess.PIPE) as proc:
    print(proc.stdout.read())