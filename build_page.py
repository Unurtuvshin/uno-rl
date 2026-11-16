"""Bundle web/engine.js + models/*.json + web/player_template.html into one file:  python build_page.py -> uno_player.html"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
models = {}
for f in sorted(glob.glob(os.path.join(HERE, "models", "*.json"))):
    m = json.load(open(f, encoding="utf8"))
    models[f"{m['algo']}_{m['opp']}_{m['episodes']}"] = m
tpl = open(os.path.join(HERE, "web", "player_template.html"), encoding="utf8").read()
eng = open(os.path.join(HERE, "web", "engine.js"), encoding="utf8").read()
esc = lambda s: s.replace("</", "<" + chr(92) + "/")           # keep "</script>" out of the embedded data
out = tpl.replace("/*ENGINE*/", "").replace("/*MODELS*/null", esc(json.dumps(models, separators=(",", ":"))))
out = out.replace("<script>\n", "<script>\n" + esc(eng) + "\n</script>\n<script>\n", 1)    # engine in its own script, before the page's
path = os.path.join(HERE, "uno_player.html")
open(path, "w", encoding="utf8").write(out)
print(f"wrote {path}: {len(models)} models, {os.path.getsize(path) / 1e6:.1f} MB")
