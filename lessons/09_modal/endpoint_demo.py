"""A tiny Modal web endpoint, to practice `modal run`, `modal serve`, and `modal deploy`.

Run these from the ML_Lab folder (notebook section 9 explains each one):

    uv run modal run lessons/09_modal/endpoint_demo.py      # call a function once, then everything shuts down
    uv run modal serve lessons/09_modal/endpoint_demo.py    # temporary URL that reloads when you save this file
    uv run modal deploy lessons/09_modal/endpoint_demo.py   # permanent URL, scales to zero when idle
    uv run modal app stop ml-lab-endpoint                   # take it down again (asks to confirm, --yes skips)

Everything here runs on CPU, so a call costs a tiny fraction of a cent.
"""

import modal

# fastapi lives in the cloud image only. You don't need it installed on your Mac to deploy this.
image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]")
app = modal.App("ml-lab-endpoint", image=image)


@app.function()
def cube(x: int) -> int:
    return x**3


@app.function()
@modal.fastapi_endpoint(method="GET")
def square(x: int = 2) -> dict:
    """GET <url>?x=5 returns {"x": 5, "square": 25, "cube": 125}."""
    # One Modal function can call another. The cube runs in its own container.
    return {"x": x, "square": x * x, "cube": cube.remote(x)}


@app.local_entrypoint()
def main():
    print("cube(3) computed in Modal's cloud:", cube.remote(3))
