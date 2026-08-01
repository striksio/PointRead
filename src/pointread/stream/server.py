import asyncio

from aiohttp import web

from pointread.stream.camera import lock, latest

PAGE = """
<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
</head>
<body style="margin:0;height:100vh;background:#111;display:flex;align-items:center;justify-content:center;overflow:hidden">
<img id="v" style="max-width:100vw;max-height:100vh;width:auto;height:auto;object-fit:contain">
<script>
const img = document.getElementById('v');
let cur = null;
const ws = new WebSocket('ws://' + location.host + '/ws');
ws.binaryType = 'arraybuffer';
ws.onmessage = (e) => {
  const blob = new Blob([e.data], {type:'image/jpeg'});
  const u = URL.createObjectURL(blob);
  img.src = u; if(cur) URL.revokeObjectURL(cur); cur = u;
};
ws.onclose = () => setTimeout(()=>location.reload(), 1000);
</script></body></html>
"""


async def index(request):
    return web.Response(text=PAGE, content_type="text/html")


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    last = None
    try:
        while not ws.closed:
            with lock:
                data = latest["jpg"]
            if data is not None and data is not last:
                await ws.send_bytes(data); last = data
            await asyncio.sleep(0.033)
    except Exception as e:
        print("ws err:", e)
    return ws


def build_app():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    return app
