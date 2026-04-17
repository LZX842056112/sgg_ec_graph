import uvicorn
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from configuration.config import WEB_STATIC_DIR
from service import ChatService
from schemas import Question

app = FastAPI()
app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")
app.add_middleware(
    SessionMiddleware,
    secret_key="ecommerce-secret-key",
    max_age=3600,
    https_only=False,
    same_site="lax"
)
service = ChatService()


@app.get("/")
def read_root():
    return RedirectResponse("/static/index.html")


@app.post("/api/chat")
async def chat(question: Question, request: Request):
    session = request.session
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_id = session["session_id"]

    return StreamingResponse(
        service.chat(question.message, session_id),
        media_type="text/plain"
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,
        timeout_graceful_shutdown=30
    )
