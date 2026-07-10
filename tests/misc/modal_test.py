import modal

app = modal.App("modal-example")

@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2

@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))

# run via
# modal run modal_test.py
# inside uv venv