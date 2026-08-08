import asyncio

from aiohttp import web

from pointread.stream.camera import lock, latest, state, signal


PAGE = """
<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
</head>
<body style="margin:0;height:100vh;background:#111;display:flex;align-items:center;justify-content:center;overflow:hidden">
<img id="v" style="max-width:100vw;max-height:100vh;width:auto;height:auto;object-fit:contain">
<audio id="silent" playsinline loop preload="auto" src="/silent.wav"></audio>
<button id="enable" style="position:fixed;inset:0;width:100%;height:100%;background:rgba(0,0,0,.8);color:#fff;font:20px monospace;border:0;z-index:20">tap to enable sound</button>
<script>
fetch('/reset');

let actx = null, raw = null, buf = null, ready = false;

fetch('/beep.wav').then(r => r.arrayBuffer()).then(b => { raw = b; }).catch(()=>{});

function decodeInto() {
  return new Promise((res, rej) => {
    const copy = raw.slice(0);
    const p = actx.decodeAudioData(copy, res, rej);
    if (p && p.then) p.then(res, rej);
  });
}

function unlock() {
  try {
    if (navigator.audioSession) { try { navigator.audioSession.type = 'playback'; } catch (e) {} }
    if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
    actx.resume();
    const s = actx.createBufferSource();
    s.buffer = actx.createBuffer(1, 1, 22050);
    s.connect(actx.destination);
    s.start(0);
    document.getElementById('silent').play().catch(()=>{});
    document.getElementById('enable').style.display = 'none';
    if (raw && !buf) {
      decodeInto().then(b => { buf = b; ready = true; }).catch(()=>{});
    }
  } catch (e) {}
}

document.getElementById('enable').addEventListener('click', unlock);
document.getElementById('enable').addEventListener('touchend', unlock, {passive:true});

document.addEventListener('visibilitychange', () => {
  if (!document.hidden && actx) {
    actx.resume();
    document.getElementById('silent').play().catch(()=>{});
  }
});

function playBeep() {
  if (!ready) return;
  if (actx.state !== 'running') actx.resume();
  const src = actx.createBufferSource();
  const g = actx.createGain();
  src.buffer = buf;
  g.gain.value = 1;
  src.connect(g).connect(actx.destination);
  src.start();
}

const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';

const ev = new WebSocket(proto + location.host + '/events');
ev.onmessage = (e) => { if (e.data === 'beep') playBeep(); };
ev.onclose = () => setTimeout(()=>location.reload(), 1500);

const img = document.getElementById('v');
let cur = null;
const ws = new WebSocket(proto + location.host + '/ws');
ws.binaryType = 'arraybuffer';
ws.onmessage = (e) => {
  const blob = new Blob([e.data], {type:'image/jpeg'});
  const u = URL.createObjectURL(blob);
  img.src = u; if (cur) URL.revokeObjectURL(cur); cur = u;
};
ws.onclose = () => setTimeout(()=>location.reload(), 1000);
</script></body></html>
"""


_clients = set()

async def events(request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    _clients.add(ws)
    print("events connected, n =", len(_clients))
    try:
        async for _ in ws:
            pass
    finally:
        _clients.discard(ws)
    return ws


async def beep_pump(app):
    while True:
        if signal["beep"]:
            signal["beep"] = False
            print("broadcast beep to", len(_clients))
            for ws in list(_clients):
                try:
                    await ws.send_str("beep")
                except Exception:
                    _clients.discard(ws)
        await asyncio.sleep(0.02)


async def _start(app):
    app["pump"] = asyncio.create_task(beep_pump(app))


async def _stop(app):
    app["pump"].cancel()


async def beep_wav(request):
    return web.FileResponse(
        "/workspace/assets/beep.wav",
        headers={"Content-Type": "audio/wav", "Cache-Control": "no-store"},
    )


async def silent_wav(request):
    return web.FileResponse(
        "/workspace/assets/silent.wav",
        headers={"Content-Type": "audio/wav", "Cache-Control": "no-store"},
    )

async def index(request):
    return web.Response(text=PAGE, content_type="text/html")


async def reset(request):
    state["reset"] = True
    return web.Response(text="ok")


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    last = None
    try:
        while not ws.closed:
            with lock:
                data = latest["jpg"]
            if data is not None and data is not last:
                await ws.send_bytes(data)
                last = data
            await asyncio.sleep(0.033)
    except Exception as e:
        print("ws err:", e)
    return ws


def build_app():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/reset", reset)
    app.router.add_get("/events", events)
    app.router.add_get("/beep.wav", beep_wav)
    app.router.add_get("/silent.wav", silent_wav)
    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
    return app
