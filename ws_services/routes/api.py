from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Header, Request, HTTPException

router = APIRouter()

agent_contexts = {}
websockets = {}

API_KEYS = {
    "strategy-manager": "secretkey1",
    "code-agent": "secretkey2",
    "backtester-agent": "secretkey3",
    "trainer-agent": "secretkey4",
}

def verify_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS.values():
        raise HTTPException(status_code=401, detail="Unauthorized")

@router.get("/context/{agent_id}")
async def get_context(agent_id: str, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    return agent_contexts.get(agent_id, {})

@router.post("/context/{agent_id}")
async def update_context(agent_id: str, request: Request, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    body = await request.json()
    agent_contexts[agent_id] = body
    return {"message": "Context updated", "context": body}

@router.post("/message/{agent_id}")
async def send_message(agent_id: str, request: Request, x_api_key: str = Header(...)):
    verify_key(x_api_key)
    body = await request.json()
    target_agent = body.get("target_agent")
    if target_agent not in websockets:
        raise HTTPException(status_code=404, detail="Target agent not connected")
    await websockets[target_agent].send_json(body)
    return {"message": "Message sent"}

@router.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    websockets[agent_id] = websocket
    print(f"🔌 {agent_id} connected via WebSocket")
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        websockets.pop(agent_id, None)
        print(f"❌ {agent_id} disconnected")