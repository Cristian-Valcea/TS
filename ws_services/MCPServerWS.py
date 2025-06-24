from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from routes import api
import time
import json

import os
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from config import CHART_DIR, TRADELOG_DIR



def log_to_file(agent_id: str, data: dict, category: str = "context"):
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    ts = datetime.utcnow().strftime("%H-%M-%S_%f")
    folder = f"logs/{agent_id}/{date_str}/{category}"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{ts}.json"

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)



app = FastAPI(title="MCP Server - Multi-Agent Communication Platform")

# Mount static for assets if needed
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Resolve paths relative to the parent directory of MCPServerMain
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
charts_dir = os.path.abspath(os.path.join(base_dir, CHART_DIR))
tradelogs_dir = os.path.abspath(os.path.join(base_dir, TRADELOG_DIR))

# ✅ Static mounts for charts and tradelogs
#charts_dir = os.path.abspath(CHART_DIR)#"../dashboard/frontend/charts")
print(f"Charts directory: {charts_dir}")
#tradelogs_dir = os.path.abspath(TRADELOG_DIR)#"../dashboard/frontend/tradelogs")
print(f"TradeLogs directory: {tradelogs_dir}")
app.mount("/charts", StaticFiles(directory=charts_dir), name="charts")
app.mount("/tradelogs", StaticFiles(directory=tradelogs_dir), name="tradelogs")

#app.mount("/static", StaticFiles(directory="static"), name="static")


# 🧩 Allow frontend to talk to server from file:// or localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← You can lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory context store
context_store: Dict[str, Dict[str, Any]] = {}

# API keys for agents
api_keys = {
    "strategy-manager": "secretkey1",
    "code-agent": "secretkey2",
    "backtester-agent": "secretkey3",
    "trainer-agent": "secretkey4",
    "trader-agent": "secretkey5"
}

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, agent_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[agent_id] = websocket

    def disconnect(self, agent_id: str):
        if agent_id in self.active_connections:
            del self.active_connections[agent_id]

    async def send_message(self, agent_id: str, message: str):
        if agent_id in self.active_connections:
            await self.active_connections[agent_id].send_text(message)

ws_manager = ConnectionManager()

# Authentication for matching agent ID
def authenticate_agent(agent_id: str, x_api_key: str = Header(...)):
    if agent_id not in api_keys or api_keys[agent_id] != x_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# Internal access for trusted agents
def authenticate_trusted(x_api_key: str = Header(...)):
    if x_api_key != api_keys["backtester-agent"]:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# Helpers
def set_context(agent_id: str, context: Dict[str, Any]):
    context_store[agent_id] = context

def get_context(agent_id: str) -> Dict[str, Any]:
    return context_store.get(agent_id, {})

@app.post("/context/{agent_id}")
async def update_context(agent_id: str, data: Dict[str, Any], auth: bool = Depends(authenticate_agent)):
    data["status"] = data.get("status", "in_progress")
    set_context(agent_id, data)
    log_to_file(agent_id, data, "context")
    print(f"[{time.strftime('%H:%M:%S')}] Context updated for '{agent_id}': {data}")
    return {"message": "Context updated", "context": data}

@app.get("/context/{agent_id}")
async def retrieve_context(agent_id: str, auth: bool = Depends(authenticate_agent)):
    context = get_context(agent_id)
    print(f"[{time.strftime('%H:%M:%S')}] Context retrieved for '{agent_id}': {context}")
    return context if context else {"message": "No context found"}

@app.get("/internal/context/{target_agent}")
async def internal_context_retrieval(target_agent: str, auth: bool = Depends(authenticate_trusted)):
    context = get_context(target_agent)
    print(f"[{time.strftime('%H:%M:%S')}] Internal context access for '{target_agent}': {context}")    
    if not context:
        return JSONResponse(status_code=200, content={"message": "No context found"})
    return context
'''
@app.get("/internal/context/{target_agent}")
async def internal_context_retrieval(target_agent: str, auth: bool = Depends(authenticate_trusted)):
    context = get_context(target_agent)
    print(f"[{time.strftime('%H:%M:%S')}] Internal context access for '{target_agent}': {context}")
    return context if context else {"message": "No context found"}
'''

@app.post("/message/{agent_id}")
async def receive_message(agent_id: str, data: Dict[str, Any], auth: bool = Depends(authenticate_agent)):
    log_to_file(agent_id, data, category="message")

    print(f"[{time.strftime('%H:%M:%S')}] Message received from '{agent_id}': {data}")
    
    target = data.get("target_agent")
    if target:
        task_context = {
            "from": agent_id,
            "task": data.get("task"),
            "payload": data.get("payload", {}),
            "status": "pending",
            "timestamp": time.time()
        }
        set_context(target, task_context)
        print(f"[{time.strftime('%H:%M:%S')}] Task routed to '{target}': {task_context}")
        if target in ws_manager.active_connections:
            await ws_manager.send_message(target, json.dumps(task_context))

    return {"message": f"Message routed to {target if target else 'none'}", "data": data}

@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    await ws_manager.connect(agent_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(agent_id)
        print(f"[{time.strftime('%H:%M:%S')}] WebSocket disconnected for '{agent_id}'")


LOG_DIR = "logs"

@app.get("/agents")
def list_agents():
    return os.listdir(LOG_DIR) if os.path.exists(LOG_DIR) else []

@app.get("/logs/{agent_id}")
def list_agent_dates(agent_id: str):
    path = os.path.join(LOG_DIR, agent_id)
    return os.listdir(path) if os.path.exists(path) else []

@app.get("/logs/{agent_id}/{date}")
def get_logs(agent_id: str, date: str):
    base = os.path.join(LOG_DIR, agent_id, date)
    results = {}
    if not os.path.exists(base):
        return results

    for category in os.listdir(base):
        folder = os.path.join(base, category)
        files = sorted(os.listdir(folder))
        results[category] = []
        for f in files:
            with open(os.path.join(folder, f)) as fp:
                data = json.load(fp)
                results[category].append({"timestamp": f[:-5], "data": data})
    return results

