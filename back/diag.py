"""Re-run analyze, keep JSON, measure talc polygon coverage, render frontend-accurate crop."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.services.analysis import analyze_sample
Image.MAX_IMAGE_PIXELS = None
COL = {"common": (34,197,94), "thin": (239,68,68), "talc": (59,130,246)}
img_path = Path("storage/4e2aa1261d8b.jpg").resolve()
res = analyze_sample(img_path)
Path("diag_result.json").write_text(json.dumps(res))

def shoelace(ring):
    x=[p[0] for p in ring]; y=[p[1] for p in ring]; s=0
    for i in range(len(ring)):
        j=(i+1)%len(ring); s+=x[i]*y[j]-x[j]*y[i]
    return abs(s)/2

cov={}; pts={}; biggest={}
for s in res["segments"]:
    a=sum(shoelace(r) for r in s["polygons"])
    cov[s["phase"]]=cov.get(s["phase"],0)+a
    p=sum(len(r) for r in s["polygons"]); pts[s["phase"]]=pts.get(s["phase"],0)+p
    biggest[s["phase"]]=max(biggest.get(s["phase"],0), a)
print("metric talcShare=%.1f sulfideShare=%.2f"%(res["talcShare"],res["sulfideShare"]))
print("polygon coverage frac (shoelace):", {k:round(v,3) for k,v in cov.items()})
print("biggest single polygon frac:", {k:round(v,3) for k,v in biggest.items()})
print("total pts by phase:", pts)

# frontend-accurate render: 4:3 canvas, object-cover crop, blue@0.6 over image
W,H = res["imageWidth"], res["imageHeight"]
base = Image.open(img_path).convert("RGB")
# emulate aspect-[4/3] cover: target canvas 1200x900
cw,ch = 1200,900
scale = max(cw/W, ch/H)
rw,rh = int(W*scale), int(H*scale)
img_scaled = base.resize((rw,rh))
left=(rw-cw)//2; top=(rh-ch)//2
canvas = img_scaled.crop((left,top,left+cw,top+ch)).convert("RGBA")
ov = Image.new("RGBA",(cw,ch),(0,0,0,0)); d=ImageDraw.Draw(ov)
# svg viewBox 0..W,0..H with slice = same cover crop; map norm->canvas
def to_canvas(x,y):
    px = x*W*scale - left; py = y*H*scale - top; return (px,py)
for s in res["segments"]:
    c = COL[s["phase"]]+(153,)  # ~0.6
    for ring in s["polygons"]:
        poly=[to_canvas(x,y) for x,y in ring]
        if len(poly)>=3: d.polygon(poly, fill=c)
out=Image.alpha_composite(canvas,ov).convert("RGB")
out.save("diag_frontend_crop.png")
print("wrote diag_frontend_crop.png", out.size)
