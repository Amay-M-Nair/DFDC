"""HTML for the demo panels (app.py). Colours come from Gradio's theme variables, so light and dark mode work."""
from html import escape

from src.demo.pipeline import MODELS, METHOD_NAMES, STEPS, heat_color
from src.explain.corpus import cards_by_id
from src.xai.gradcam import ATOMIC_BACKGROUND, BACKGROUND, PROMINENT

CSS = """
:root {--dd-fake: #e5484d; --dd-real: #30a46c; --dd-accent: #3b82f6; --dd-warn: #f5a524; --dd-grey: #8e8c99;}
.html-container {padding: 0 !important;}  /* every gr.HTML is one of ours: align the cards with the image blocks */
.dd-muted {color: var(--body-text-color-subdued); font-size: 0.85rem;}
.dd-mono {font-family: var(--font-mono);}
.dd-num {font-variant-numeric: tabular-nums;}

/* header and notes */
.dd-header {display: flex; align-items: center; gap: 14px; padding: 6px 0 4px;}
.dd-logo {flex: none; width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center; color: #fff;
          background: linear-gradient(135deg, #3b82f6, #6366f1); box-shadow: 0 4px 14px rgba(59, 130, 246, 0.28);}
.dd-header h1 {margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2;}
.dd-header p {margin: 2px 0 0; color: var(--body-text-color-subdued); font-size: 0.92rem;}
.dd-banner {display: flex; gap: 10px; align-items: flex-start; padding: 10px 14px; border-radius: 10px;
            font-size: 0.9rem; line-height: 1.5; border: 1px solid color-mix(in srgb, var(--dd-accent) 30%, transparent);
            background: color-mix(in srgb, var(--dd-accent) 7%, transparent);}
.dd-banner svg {flex: none; margin-top: 2px; color: var(--dd-accent);}
.dd-note {max-width: 760px; margin: 10px auto 0; text-align: center; font-size: 0.82rem; line-height: 1.5;
          color: var(--body-text-color-subdued);}

/* cards, titles, chips */
.dd-card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 16px 18px;
          background: var(--block-background-fill);}
.dd-title {display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap;
           font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
           color: var(--body-text-color-subdued); margin-bottom: 10px;}
.dd-meta {text-transform: none; letter-spacing: 0; font-weight: 500;}
.dd-subhead {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
             color: var(--body-text-color-subdued); margin: 16px 0 6px; padding-top: 12px;
             border-top: 1px solid var(--border-color-primary);}
.dd-empty {border-style: dashed; background: transparent;}
.dd-empty .dd-title {margin-bottom: 6px;}
.dd-msg {display: flex; align-items: center; gap: 8px; color: var(--body-text-color-subdued); font-size: 0.88rem;
         line-height: 1.5;}
.dd-spin {flex: none; width: 14px; height: 14px; border-radius: 50%; border: 2px solid var(--border-color-primary);
          border-top-color: var(--dd-accent); animation: dd-rot 0.8s linear infinite;}
@keyframes dd-rot {to {transform: rotate(360deg);}}
.dd-chip {display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 0.76rem; font-weight: 600;
          line-height: 1.6; border: 1px solid var(--border-color-primary); margin: 2px 4px 2px 0; white-space: nowrap;}
.dd-chip.tone {color: var(--c); border-color: color-mix(in srgb, var(--c) 45%, transparent);
               background: color-mix(in srgb, var(--c) 10%, transparent);}
.dd-ok {color: var(--dd-real); font-weight: 700;} .dd-bad {color: var(--dd-fake); font-weight: 700;}

/* progress strip */
.dd-steps {display: flex; align-items: flex-start; gap: 0; margin: 6px 0 2px;}
.dd-step {flex: 1; text-align: center; position: relative; min-width: 0;}
.dd-step:not(:first-child)::before {content: ""; position: absolute; top: 13px; right: 50%; width: 100%; height: 2px;
          background: var(--border-color-primary); z-index: 0;}
.dd-step.done:not(:first-child)::before, .dd-step.active:not(:first-child)::before,
.dd-step.failed:not(:first-child)::before {background: var(--dd-real);}
.dd-dot {position: relative; z-index: 1; display: inline-flex; width: 28px; height: 28px; border-radius: 50%;
         align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700;
         background: var(--block-background-fill); border: 2px solid var(--border-color-primary);
         color: var(--body-text-color-subdued);}
.dd-step.done .dd-dot {background: var(--dd-real); border-color: var(--dd-real); color: #fff;}
.dd-step.active .dd-dot {border-color: var(--dd-accent); color: var(--dd-accent); animation: dd-pulse 1.1s ease-in-out infinite;}
.dd-step.failed .dd-dot {background: var(--dd-fake); border-color: var(--dd-fake); color: #fff;}
.dd-name {display: block; font-size: 0.78rem; font-weight: 600; margin-top: 4px;}
.dd-time {display: block; font-size: 0.7rem; color: var(--body-text-color-subdued); min-height: 1em;}
.dd-step.pending .dd-name {color: var(--body-text-color-subdued); font-weight: 500;}
.dd-error {margin-top: 8px; padding: 8px 10px; border-radius: 8px; color: var(--dd-fake); font-size: 0.82rem;
           background: color-mix(in srgb, var(--dd-fake) 9%, transparent);}
@keyframes dd-pulse {0%, 100% {box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.45);}
                     50% {box-shadow: 0 0 0 6px rgba(59, 130, 246, 0);}}

/* verdict */
.dd-vcard {display: grid; grid-template-columns: minmax(150px, auto) minmax(0, 1fr); gap: 10px 32px; align-items: center;
           border-left: 4px solid var(--c);
           background: linear-gradient(90deg, color-mix(in srgb, var(--c) 11%, transparent), transparent 55%),
                       var(--block-background-fill);}
.dd-verdict {font-size: 2.4rem; font-weight: 800; letter-spacing: 0.02em; line-height: 1; color: var(--c);
             margin: 2px 0 8px;}
.dd-prob {font-size: 0.95rem;}
.dd-prob b {font-size: 1.15rem;}
.dd-tick {display: inline-block; width: 2px; height: 11px; margin: 0 3px -1px 6px; background: var(--body-text-color);}
.dd-gauge {position: relative; height: 10px; border-radius: 999px; margin: 8px 0 4px;
           background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);}
.dd-gauge > span {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: var(--c);}
.dd-gauge > i {position: absolute; top: -5px; width: 2px; height: 18px; border-radius: 1px; background: var(--body-text-color);}
.dd-scale {display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--body-text-color-subdued);}
.dd-truth {grid-column: 1 / -1; display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; font-size: 0.9rem;
           padding-top: 10px; border-top: 1px dashed var(--border-color-primary);}
.dd-vcard > .dd-muted {grid-column: 1 / -1;}
@media (max-width: 640px) {.dd-vcard {grid-template-columns: 1fr;}}

/* region heat */
.dd-ramp {display: flex; align-items: center; gap: 8px; font-size: 0.72rem; color: var(--body-text-color-subdued);
          margin: 0 0 10px;}
.dd-ramp > span {flex: 1; height: 6px; border-radius: 999px;}
.dd-region {display: grid; grid-template-columns: 10px minmax(0, 7.4em) minmax(36px, 1fr) auto; gap: 8px;
            align-items: center; padding: 3px 0; font-size: 0.85rem;}
.dd-region > b {font-weight: 600; line-height: 1.25;}
.dd-region.cold {opacity: 0.72;}
.dd-swatch {width: 10px; height: 10px; border-radius: 3px;}
.dd-swatch.hollow {background: transparent !important; border: 1.5px dashed var(--body-text-color-subdued);}
.dd-track {height: 6px; border-radius: 999px; background: var(--background-fill-secondary); overflow: hidden;}
.dd-track > span {display: block; height: 100%; border-radius: 999px;}
.dd-val {text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;}
.dd-divider {display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 0.72rem;
             color: var(--body-text-color-subdued);}
.dd-divider::before, .dd-divider::after {content: ""; flex: 1; border-top: 1px dashed var(--border-color-primary);}
.dd-foot {margin-top: 10px; font-size: 0.8rem; color: var(--body-text-color-subdued);}

/* explanation */
.dd-expl {font-size: 1.04rem; line-height: 1.65; margin: 2px 0 4px;}
.dd-finding {display: grid; grid-template-columns: 10px minmax(0, 1fr) auto; gap: 10px; align-items: start;
             padding: 7px 0; font-size: 0.86rem;}
.dd-finding + .dd-finding {border-top: 1px solid var(--border-color-primary);}
.dd-finding .dd-swatch {margin-top: 5px;}
.dd-finding .dd-muted {font-size: 0.8rem; margin-top: 2px;}
.dd-sources {display: flex; flex-wrap: wrap; align-items: center; gap: 2px 6px; margin-top: 12px; font-size: 0.82rem;}
.dd-checks {display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;}
.dd-pill {display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 8px; font-size: 0.78rem;
          font-weight: 500; background: var(--background-fill-secondary); color: var(--body-text-color-subdued);}
.dd-pill.ok {color: var(--dd-real); background: color-mix(in srgb, var(--dd-real) 11%, transparent);}
.dd-pill.bad {color: var(--dd-fake); background: color-mix(in srgb, var(--dd-fake) 11%, transparent);}

/* evidence cards and summary */
.dd-cards {display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px;}
.dd-ev {display: flex; flex-direction: column; border: 1px solid var(--border-color-primary); border-radius: 10px;
        padding: 12px 14px; font-size: 0.86rem; line-height: 1.5; background: var(--background-fill-primary);}
.dd-ev.dim {opacity: 0.66;}
.dd-ev.cited {border-color: var(--dd-accent);
              box-shadow: 0 0 0 1px var(--dd-accent), 0 4px 14px rgba(59, 130, 246, 0.14);}
.dd-ev-head {display: flex; justify-content: space-between; align-items: center; gap: 6px;}
.dd-ev-id {margin: 8px 0 4px; font-size: 0.84rem;}
.dd-ev .dd-src {margin-top: auto; padding-top: 8px; font-style: italic; color: var(--body-text-color-subdued);
                font-size: 0.78rem;}
.dd-summary {font-size: 0.84rem; line-height: 1.6; padding: 12px 14px; border-radius: 8px; white-space: pre-wrap;
             background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);}
"""

KIND_STYLE = {"artifact": ("Artefact", "var(--dd-fake)"), "scope": ("Scope", "var(--dd-warn)"),
              "limitation": ("Limitation", "var(--dd-grey)"), "authentic": ("Authentic cue", "var(--dd-real)")}
TONE = {"fake": "var(--dd-fake)", "real": "var(--dd-real)"}
LOGO = ("<svg viewBox='0 0 24 24' width='24' height='24' fill='none' stroke='currentColor' stroke-width='2' "
        "stroke-linecap='round' stroke-linejoin='round'><path d='M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2"
        "M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2'/><circle cx='12' cy='10' r='3'/>"
        "<path d='M7 18c1-2.5 3-3.5 5-3.5s4 1 5 3.5'/></svg>")
INFO = ("<svg viewBox='0 0 24 24' width='16' height='16' fill='none' stroke='currentColor' stroke-width='2' "
        "stroke-linecap='round'><circle cx='12' cy='12' r='9'/><path d='M12 11v5M12 8h.01'/></svg>")


def _rgb(c):
    return f"rgb{tuple(c)}"


def nice(label: str) -> str:
    return label.replace("_", " ")


def chip(text: str, tone: str = None, cls: str = "") -> str:
    """tone: a CSS colour that tints the chip; none = neutral."""
    if tone:
        return f"<span class='dd-chip tone {cls}' style='--c:{tone}'>{text}</span>"
    return f"<span class='dd-chip {cls}'>{text}</span>"


def pill(text: str, ok: bool = None) -> str:
    cls = "" if ok is None else " ok" if ok else " bad"
    return f"<span class='dd-pill{cls}'>{text}</span>"


def header() -> str:
    return (f"<div class='dd-header'><div class='dd-logo'>{LOGO}</div><div><h1>Explainable Deepfake Detector</h1>"
            "<p>EfficientNet-B0 detector · Grad-CAM++ evidence · explanation grounded in forensic literature</p>"
            "</div></div>")


def banner(html: str) -> str:
    return f"<div class='dd-banner'>{INFO}<div>{html}</div></div>"


def note(html: str) -> str:
    return f"<div class='dd-note'>{html}</div>"


def progress(finished: int = -1, error: str = "", timings: dict = None, running: bool = False) -> str:
    """finished: index of the last completed step (-1 = none)."""
    timings = timings or {}
    parts = []
    for i, name in enumerate(STEPS):
        if i <= finished:
            state, mark = "done", "✓"
        elif i == finished + 1 and error:
            state, mark = "failed", "✕"
        elif i == finished + 1 and running:
            state, mark = "active", str(i + 1)
        else:
            state, mark = "pending", str(i + 1)
        t = f"{timings[name]:.1f} s" if name in timings and state == "done" else ""
        parts.append(f"<div class='dd-step {state}'><span class='dd-dot'>{mark}</span>"
                     f"<span class='dd-name'>{name}</span><span class='dd-time'>{t}</span></div>")
    err = f"<div class='dd-error'>{escape(error)}</div>" if error else ""
    return f"<div class='dd-steps'>{''.join(parts)}</div>{err}"


def placeholder(title: str, text: str, busy: bool = False) -> str:
    spin = "<span class='dd-spin'></span>" if busy else ""
    return (f"<div class='dd-card dd-empty'><div class='dd-title'><span>{escape(title)}</span></div>"
            f"<div class='dd-msg'>{spin}<span>{escape(text)}</span></div></div>")


def verdict(res, prompt: str = "") -> str:
    """prompt: what the empty panel says; none = analysing."""
    if res is None or res.prob_fake is None:
        return placeholder("Verdict", prompt or "Analysing…", busy=not prompt)
    tone = TONE[res.label]
    chips = chip(f"{res.confidence} confidence")
    if res.cached:
        chips += chip("cached result")
    no_face = ""
    if res.face_found is False:
        no_face = "<div class='dd-muted'>No face detected: the whole image was analysed.</div>"
    truth = ""
    if res.truth:
        t = res.truth
        right = (res.label == "fake") == (t["label"] == "fake")
        truth = (f"<div class='dd-truth'><span class='dd-muted'>Ground truth</span> <span><b>{t['label'].upper()}</b> · "
                 f"{escape(METHOD_NAMES.get(t['manipulation'], t['manipulation']))} "
                 f"<span class='dd-muted'>({escape(t['dataset'])})</span></span>"
                 f"{pill('✓ detector correct', True) if right else pill('✕ detector wrong', False)}</div>")
    return (f"<div class='dd-card dd-vcard' style='--c:{tone}'><div><div class='dd-title'>Verdict</div>"
            f"<div class='dd-verdict'>{res.label.upper()}</div>{chips}</div>"
            f"<div><div class='dd-prob dd-num'>P(fake) <b>{res.prob_fake:.3f}</b> <span class='dd-muted'>"
            f"<span class='dd-tick'></span>decision threshold {res.threshold:.3f}</span></div>"
            f"<div class='dd-gauge'><span style='width:{res.prob_fake * 100:.1f}%'></span>"
            f"<i style='left:calc({res.threshold * 100:.1f}% - 1px)' title='decision threshold'></i></div>"
            f"<div class='dd-scale'><span>0 · real</span><span>fake · 1</span></div></div>{no_face}{truth}</div>")


def heat_ramp() -> str:
    """The region colours' scale (heat_color), cold to hottest."""
    stops = ", ".join(_rgb(heat_color(i / 6, 1)) for i in range(7))
    return (f"<div class='dd-ramp'>cold<span style='background:linear-gradient(90deg, {stops})'></span>"
            "hottest</div>")


def regions(res) -> str:
    if res is None or not res.top_regions:
        return placeholder("Where the detector looked",
                           "Every facial region, ranked by how much of the detector's evidence it holds, will appear here.")
    hottest = max(c for _, c, _ in res.top_regions) or 1
    rows, n_prominent = [], sum(c >= PROMINENT * hottest for _, c, _ in res.top_regions)
    for i, (name, conc, share) in enumerate(res.top_regions):
        c = _rgb(heat_color(conc, hottest))
        unboxed = name in (BACKGROUND, ATOMIC_BACKGROUND)
        if i == n_prominent:
            rows.append(f"<div class='dd-divider'>↑ prominent: ≥ {PROMINENT:.0%} of the hottest</div>")
        rows.append(f"<div class='dd-region{' cold' if i >= n_prominent else ''}'>"
                    f"<span class='dd-swatch{' hollow' if unboxed else ''}' style='background:{c}'></span>"
                    f"<b>{escape(name)}</b>"
                    f"<span class='dd-track'><span style='width:{conc / hottest * 100:.0f}%;background:{c}'></span></span>"
                    f"<span class='dd-val'>{conc:.1f}× <span class='dd-muted'>· {share:.0%}</span></span></div>")
    names = [name for name, _, _ in res.top_regions]
    src = {"mesh": "" if "teeth" in names else "Mouth closed, so there is no teeth region.",
           "landmarks": "Region boxes from YuNet's 5 landmarks (MediaPipe found no face), so teeth is not placed.",
           }.get(res.region_source, "Region boxes from a fixed grid (no facial landmarks found).")
    if any(name in (BACKGROUND, ATOMIC_BACKGROUND) for name in names):
        src += " Dashed swatch: no box on the image."
    src = f"<div class='dd-foot'>{src.strip()}</div>" if src else ""
    return (f"<div class='dd-card'><div class='dd-title'><span>What made it look "
            f"<span style='color:{TONE[res.label]}'>{res.label}</span></span>"
            f"<span class='dd-meta'>heat × image avg · share</span></div>{heat_ramp()}{''.join(rows)}{src}</div>")


def summary(res) -> str:
    if res is None or not res.summary:
        return "<div class='dd-muted'>The evidence summary sent to the LLM will appear here.</div>"
    return (f"<div class='dd-summary dd-mono'>{escape(res.summary)}</div>"
            "<div class='dd-muted' style='margin-top:6px'>Filled from fixed templates: the LLM reads this text "
            "and the evidence cards, never the image.</div>")


def explanation(res, waiting: bool = False) -> str:
    title = "Grounded explanation"
    if res is None or (res.answer is None and not waiting and not res.cards):
        return placeholder(title, "One finding per prominent region, each backed by a cited evidence card, "
                                  "will appear here.")
    model = MODELS.get(res.model, res.model)
    if res.answer is None:
        text = (f"Asking {model}…" if waiting else
                f"No stored answer from {model} for this image." if res.cached else "No explanation.")
        return placeholder(title, text, busy=waiting)
    a = res.answer
    if not a.schema_ok:
        return placeholder(title, f"{model}'s reply could not be used: {(a.error or a.raw)[:200]}")
    hits, invented = set(res.check.get("hits", [])), set(res.check.get("invented", []))
    cites = "".join(chip(escape(c), "var(--dd-accent)" if c in hits else "var(--dd-fake)", "dd-mono")
                    for c in dict.fromkeys(a.citations))
    # one row per finding; answers saved before findings existed become a single row
    rows = res.check.get("findings") or [{"region": a.claimed_region, "artifact_type": a.artifact_type,
                                          "citations": a.citations, "rank": res.check.get("region_rank"),
                                          "heat": res.check.get("region_heat", 0.0), "prominent": None,
                                          "backed": res.check.get("artifact_backed")}]
    hottest = max((c for _, c, _ in res.top_regions), default=1) or 1
    lines = []
    for f in rows:
        artefact = (chip("no artefact: judged authentic", "var(--dd-real)") if f["artifact_type"] == "none"
                    else chip(escape(nice(f["artifact_type"])), "var(--dd-fake)"))
        where = (f"#{f['rank']} of {len(res.top_regions)} by heat · {f['heat']:.1f}×" if f["rank"]
                 else "not one of this face's regions")
        if f["prominent"] is False:
            where += " · <span class='dd-bad'>not prominent</span>"
        mark = ("" if f["backed"] is None else "<span class='dd-ok'>✓ backed</span>" if f["backed"]
                else "<span class='dd-bad'>✕ not backed</span>")
        own = ", ".join(escape(c) for c in f["citations"]) or "none"
        lines.append(f"<div class='dd-finding'><span class='dd-swatch' style='background:"
                     f"{_rgb(heat_color(f['heat'], hottest))}'></span><div><b>{escape(f['region'])}</b> {artefact}"
                     f"<div class='dd-muted'>{where} · cites <span class='dd-mono'>{own}</span></div></div>"
                     f"<span>{mark}</span></div>")
    checks = []
    if res.check.get("prominent"):
        covered, prominent = res.check.get("covered", []), res.check["prominent"]
        ok = len(covered) == len(prominent)
        checks.append(pill(f"{'✓' if ok else '✕'} covers {len(covered)} of {len(prominent)} prominent regions", ok))
    if any(f["backed"] is not None for f in rows):
        backed = sum(bool(f["backed"]) for f in rows)
        ok = backed == len(rows)
        checks.append(pill(f"{'✓' if ok else '✕'} {backed} of {len(rows)} finding(s) backed by a cited card", ok))
    checks.append(pill(f"cites {len(hits)} of the {len(res.cards)} cards it was given"))
    if invented:
        checks.append(pill(f"✕ {len(invented)} id(s) not given to it", False))
    latency = f" · {res.timings['Explain']:.1f} s" if "Explain" in res.timings else ""
    return (f"<div class='dd-card'><div class='dd-title'><span>{title}</span>"
            f"<span class='dd-meta'>{escape(model)}{latency}</span></div>"
            f"<div class='dd-expl'>{escape(a.explanation)}</div>"
            f"<div class='dd-sources'><span class='dd-muted'>Sources cited</span>{cites or 'none'}</div>"
            f"<div class='dd-subhead'>Findings · {len(rows)}</div>{''.join(lines)}"
            f"<div class='dd-checks'>{''.join(checks)}</div></div>")


def cards(res) -> str:
    if res is None or not res.cards:
        return placeholder("Evidence cards", "The evidence cards retrieved from the forensic-literature corpus "
                                             "will appear here.")
    by_id, cited = cards_by_id(), set(res.check.get("hits", [])) if res.answer is not None else set()
    out = []
    for i, cid in enumerate(res.cards, 1):
        d = by_id.get(cid)
        if d is None:
            continue
        kind, color = KIND_STYLE.get(d.kind, (d.kind, "var(--dd-grey)"))
        about = f" <span class='dd-muted'>· {escape(nice(d.artifact_type))}</span>" if d.kind in ("artifact", "scope") else ""
        badge = (chip("cited ✓", "var(--dd-accent)") if cid in cited
                 else chip("offered") if res.answer is not None else "")
        state = " cited" if cid in cited else " dim" if res.answer is not None else ""
        out.append(f"<div class='dd-ev{state}'><div class='dd-ev-head'>{chip(kind, color)}{badge}</div>"
                   f"<div class='dd-ev-id'><b class='dd-mono'>{i}. {escape(cid)}</b>{about}</div>"
                   f"<div>{escape(d.text)}</div><div class='dd-src'>{escape(d.citation)}</div></div>")
    n_cited = len(cited & set(res.cards))
    meta = f"{len(res.cards)} offered" + (f" · {n_cited} cited" if res.answer is not None else "")
    return (f"<div class='dd-card'><div class='dd-title'><span>Evidence cards retrieved from the corpus</span>"
            f"<span class='dd-meta'>{meta}</span></div><div class='dd-cards'>{''.join(out)}</div></div>")
