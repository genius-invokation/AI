from ai import HeuristicPlayer, LLMPlayer
from gitcg.proto import Request as RpcRequest, Response as RpcResponse, Notification
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import FileResponse, Response
import asyncio

players = [HeuristicPlayer(0), HeuristicPlayer(1)]

app = FastAPI()

@app.post("/notify/{who}")
async def on_notify(who: int, request: Request):
    body = await request.body()
    players[who].on_notify(Notification.FromString(body))

@app.post("/rpc/{who}")
async def on_rpc(who: int, request: Request):
    rpc_request = await request.body()
    response = players[who]._on_rpc(RpcRequest.FromString(rpc_request))
    # 有点“太快了看不清楚”……加上 sleep
    await asyncio.sleep(1.5)
    return Response(response.SerializeToString())


@app.get("/")
def index():
    return FileResponse("index.html")

# 在文件末尾添加
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)