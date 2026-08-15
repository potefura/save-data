import io, math, os, secrets, time
import numpy as np
import requests
from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps, UnidentifiedImageError

TEMPLATE_URL = "http://verify.potefura.jp:3000/template.png"
SESSION_TTL = int(os.environ.get("CAPTCHA_SESSION_TTL", 300))
TOKEN_TTL = int(os.environ.get("CAPTCHA_TOKEN_TTL", 120))

W, H = 320, 180          # 作業キャンバスサイズ（テンプレートはここへ自動リサイズ）
SIZE, R, KNOB = 52, 10, 11
PAD = KNOB + 3
TOL = 6                  # 正解判定の許容誤差(px)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers="*", methods="*")

_tmpl = {"img": None}
_sessions = {}
_tokens = {}


# ---------- パズル画像生成（簡略版）----------------------------------------

def _arc(cx, cy, r, a0, a1, n=16):
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def _piece_mask():
    """角丸四角＋上辺に凸ノブ＋右辺に凹ノッチ、のアンチエイリアス済みマスク。"""
    full = SIZE + PAD * 2
    ss = full * 4
    img = Image.new("L", (ss, ss), 0)
    d = ImageDraw.Draw(img)
    pts = [(R, 0), (SIZE * 0.32, 0)] + _arc(SIZE * 0.5, 0, KNOB, math.pi, 0) + [(SIZE - R, 0)] \
        + _arc(SIZE - R, R, R, -math.pi / 2, 0) + [(SIZE, SIZE * 0.32)] \
        + _arc(SIZE, SIZE * 0.5, KNOB, -math.pi / 2, math.pi / 2) + [(SIZE, SIZE - R)] \
        + _arc(SIZE - R, SIZE - R, R, 0, math.pi / 2) + [(R, SIZE)] \
        + _arc(R, SIZE - R, R, math.pi / 2, math.pi) + [(0, R)] \
        + _arc(R, R, R, math.pi, math.pi * 1.5)
    d.polygon([((x + PAD) * 4, (y + PAD) * 4) for x, y in pts], fill=255)
    return img.resize((full, full), Image.LANCZOS)


def _prepare_template(im):
    im = ImageOps.exif_transpose(im).convert("RGB")
    sr, dr = im.width / im.height, W / H
    if sr > dr:
        nh, nw = H, round(H * sr)
    else:
        nw, nh = W, round(W / sr)
    im = im.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - W) // 2, (nh - H) // 2
    return im.crop((l, t, l + W, t + H))


def _build_challenge(tmpl):
    mask = _piece_mask()
    full = SIZE + PAD * 2
    tx = np.random.randint(SIZE + 14, W - full - 14) + PAD
    ty = np.random.randint(14, H - full - 14) + PAD
    box = (tx - PAD, ty - PAD, tx - PAD + full, ty - PAD + full)

    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    band = np.asarray(ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(7))), dtype=np.float32) / 255.0
    xx, yy = np.meshgrid(np.linspace(-1, 1, full), np.linspace(-1, 1, full))
    grad = (xx + yy); grad = (grad - grad.min()) / (grad.max() - grad.min())

    # ピース（切り出し + ベベル）
    region = tmpl.crop(box).convert("RGBA")
    piece = np.asarray(region).astype(np.float32)
    piece[..., :3] += (np.clip((grad - 0.5) * 2, 0, 1) * band)[..., None] * 70
    piece[..., :3] -= (np.clip((0.5 - grad) * 2, 0, 1) * band)[..., None] * 70
    piece[..., :3] = np.clip(piece[..., :3], 0, 255)
    piece_img = Image.fromarray(piece.astype(np.uint8), "RGBA")
    piece_img.putalpha(mask)

    # 背景（穴 + 内側シャドウ）
    large = tmpl.convert("RGBA").copy()
    hole = np.asarray(large.crop(box)).astype(np.float32)
    hole[..., :3] *= (1 - mask_np[..., None] * (0.42 + 0.18 * (1 - grad[..., None])))
    hole[..., :3] -= (band * (1 - grad) * 0.55)[..., None] * 255
    hole[..., :3] = np.clip(hole[..., :3], 0, 255)
    large.paste(Image.fromarray(hole.astype(np.uint8), "RGBA"), box)

    large_buf, small_buf = io.BytesIO(), io.BytesIO()
    large.convert("RGB").save(large_buf, "PNG")
    piece_img.save(small_buf, "PNG")
    return large_buf.getvalue(), small_buf.getvalue(), tx - PAD, ty - PAD


# ---------- セッション管理 --------------------------------------------------

def _get_template(force=False):
    # すでに画像を取得済みで、force=True でなければキャッシュを返す
    if _tmpl["img"] is not None and not force:
        return _tmpl["img"]

    # 1. ローカルのファイルを直接読み込む
    file_path = "./template.png"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"テンプレートファイルが見つかりません: {file_path}")

    # 2. 画像データの解析とテンプレート加工
    try:
        img = Image.open(file_path)
        processed_img = _prepare_template(img)
        _tmpl["img"] = processed_img
        return processed_img
    except Exception as e:
        raise ValueError(f"画像データの解析に失敗しました: {str(e)}")

def _purge():
    now = time.time()
    for k in [k for k, v in _sessions.items() if v["exp"] < now]:
        _sessions.pop(k, None)
    for k in [k for k, v in _tokens.items() if v["exp"] < now]:
        _tokens.pop(k, None)


def _session_or_404(sid):
    _purge()
    s = _sessions.get(sid)
    if not s:
        abort(404, description="captcha session not found or expired")
    return s


# ---------- ルーティング -----------------------------------------------------

@app.get("/captcha/init")
def captcha_init():
    try:
        img = _get_template(force=True)
    except requests.RequestException as e:
        return jsonify(error="template fetch failed", detail=str(e)), 502
    return jsonify(status="ok", template_url=TEMPLATE_URL, canvas_w=img.width, canvas_h=img.height)


@app.get("/captcha/start")
def captcha_start():
    try:
        tmpl = _get_template()
        large, small, tx, ty = _build_challenge(tmpl)
    except requests.RequestException as e:
        return jsonify(error="template fetch failed", detail=str(e)), 502

    sid = secrets.token_urlsafe(16)
    now = time.time()
    _sessions[sid] = {"tx": tx, "ty": ty, "large": large, "small": small, "used": False, "exp": now + SESSION_TTL}

    base = request.host_url.rstrip("/")
    return jsonify(
        id=sid,
        image_large_url=f"{base}/captcha/image/{sid}",
        image_small_url=f"{base}/captcha/image/small/{sid}",
        piece_y=ty, canvas_w=W, canvas_h=H, piece_size=SIZE, piece_pad=PAD,
        tolerance_px=TOL, expires_in=SESSION_TTL,
    )


@app.get("/captcha/image/<sid>")
def captcha_image_large(sid):
    return send_file(io.BytesIO(_session_or_404(sid)["large"]), mimetype="image/png", max_age=0)


@app.get("/captcha/image/small/<sid>")
def captcha_image_small(sid):
    return send_file(io.BytesIO(_session_or_404(sid)["small"]), mimetype="image/png", max_age=0)


def _judge(sid, sess, x_raw):
    if sess["used"]:
        abort(409, description="this captcha has already been answered")
    sess["used"] = True
    try:
        x = float(x_raw)
    except (TypeError, ValueError):
        abort(400, description="x must be a number")
    diff = abs(x - sess["tx"])
    ok = diff <= TOL
    out = {"success": ok, "diff": round(diff, 2)}
    if ok:
        token = secrets.token_urlsafe(24)
        _tokens[token] = {"sid": sid, "exp": time.time() + TOKEN_TTL, "used": False}
        out["token"] = token
    return out


@app.get("/captcha/verified/answer/<sid>")
def captcha_answer(sid):
    sess = _session_or_404(sid)
    x = request.args.get("x")
    if x is None:
        abort(400, description="missing x query param")
    return jsonify(_judge(sid, sess, x))


@app.get("/captcha/solved/<sid>/<answer>")
def captcha_solved(sid, answer):
    sess = _session_or_404(sid)
    return jsonify(_judge(sid, sess, answer))


@app.get("/captcha/verify/<sid>")
def captcha_verify(sid):
    _purge()
    token = request.args.get("token", "")
    e = _tokens.get(token)
    if not e or e["sid"] != sid:
        return jsonify(verified=False, reason="invalid or expired token"), 400
    if e["used"]:
        return jsonify(verified=False, reason="token already used"), 409
    e["used"] = True
    _tokens.pop(token, None)
    return jsonify(verified=True)


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
@app.errorhandler(502)
def _err(e):
    return jsonify(error=getattr(e, "description", str(e))), e.code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
