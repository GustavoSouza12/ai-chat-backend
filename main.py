from fastapi import FastAPI

app = FastAPI()

list = {
    'oi': "oi1",
    'oi2': "oi2"
}

@app.get("/")
def root():
    return list["oi"]

