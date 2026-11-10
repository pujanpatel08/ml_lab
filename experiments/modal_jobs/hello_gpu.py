import modal

app = modal.App("hello-gpu")

@app.function(gpu="T4")
def check_gpu():
    import subprocess
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout

@app.local_entrypoint()
def main():
    print(check_gpu.remote())