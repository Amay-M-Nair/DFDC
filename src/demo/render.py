"""HTML for the demo panels (app.py). Colours come from Gradio's theme variables, so light and dark mode work."""
from html import escape

from src.demo.pipeline import MODELS, METHOD_NAMES, STEPS, TOP_COLORS
from src.explain.corpus import cards_by_id
from src.xai.gradcam import BACKGROUND

CSS = """
.dd-header h1 {margin: 0; font-size: 1.6rem; letter-spacing: -0.01em;}
.dd-header p {margin: 4px 0 0; color: var(--body-text-color-subdued); font-size: 0.95rem;}
.dd-muted {color: var(--body-text-color-subdued); font-size: 0.85rem;}
.dd-card {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 14px 16px;
          background: var(--block-background-fill);}
.dd-title {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
           color: var(--body-text-color-subdued); margin-bottom: 8px;}
.dd-chip {display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
          border: 1px solid var(--border-color-primary); margin: 2px 4px 2px 0; white-space: nowrap;}
.dd-mono {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;}

/* progress strip */
.dd-steps {display: flex; align-items: flex-start; gap: 0; margin: 4px 0 2px;}
.dd-step {flex: 1; text-align: center; position: relative; min-width: 0;}
.dd-step:not(:first-child)::before {content: ""; position: absolute; top: 13px; right: 50%; width: 100%; height: 2px;
          background: var(--border-color-primary); z-index: 0;}
.dd-step.done:not(:first-child)::before, .dd-step.active:not(:first-child)::before,
.dd-step.failed:not(:first-child)::before {background: #30a46c;}
.dd-dot {position: relative; z-index: 1; display: inline-flex; width: 28px; height: 28px; border-radius: 50%;
         align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700;
         background: var(--block-background-fill); border: 2px solid var(--border-color-primary);
         color: var(--body-text-color-subdued);}
.dd-step.done .dd-dot {background: #30a46c; border-color: #30a46c; color: #fff;}
.dd-step.active .dd-dot {border-color: #3b82f6; color: #3b82f6; animation: dd-pulse 1.1s ease-in-out infinite;}
.dd-step.failed .dd-dot {background: #e5484d; border-color: #e5484d; color: #fff;}
.dd-name {display: block; font-size: 0.78rem; font-weight: 600; margin-top: 4px;}
.dd-time {display: block; font-size: 0.7rem; color: var(--body-text-color-subdued); min-height: 1em;}
.dd-step.pending .dd-name {color: var(--body-text-color-subdued); font-weight: 500;}
.dd-error {margin-top: 8px; color: #e5484d; font-size: 0.82rem;}
@keyframes dd-pulse {0%, 100% {box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.45);}
                     50% {box-shadow: 0 0 0 6px rgba(59, 130, 246, 0);}}

/* verdict */
.dd-verdict {font-size: 2.1rem; font-weight: 800; letter-spacing: 0.02em; line-height: 1.1;}
.dd-fake {color: #e5484d;} .dd-real {color: #30a46c;}
.dd-gauge {position: relative; height: 8px; border-radius: 4px; background: var(--border-color-primary); margin: 10px 0 4px;}
.dd-gauge > span {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px;}
.dd-gauge > i {position: absolute; top: -4px; width: 2px; height: 16px; background: var(--body-text-color);}
.dd-truth {margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--border-color-primary); font-size: 0.9rem;}

/* top-3 regions */
.dd-region {display: grid; grid-template-columns: 26px 1fr; gap: 8px; align-items: center; margin: 8px 0;}
.dd-num {width: 22px; height: 22px; border-radius: 5px; color: #fff; font-weight: 700; font-size: 0.8rem;
         display: inline-flex; align-items: center; justify-content: center;}
.dd-bar {height: 6px; border-radius: 3px; margin-top: 4px;}

/* explanation and cards */
.dd-expl {font-size: 1.02rem; line-height: 1.6; margin: 10px 0;}
.dd-check {font-size: 0.82rem; color: var(--body-text-color-subdued); margin-top: 8px;}
.dd-ok {color: #30a46c; font-weight: 700;} .dd-bad {color: #e5484d; font-weight: 700;}
.dd-cards {display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px;}
.dd-ev {border: 1px solid var(--border-color-primary); border-radius: 9px; padding: 10px 12px; font-size: 0.86rem;
        line-height: 1.45; opacity: 0.62; background: var(--block-background-fill);}
.dd-ev.cited {opacity: 1; border: 2px solid #3b82f6; box-shadow: 0 1px 6px rgba(59, 130, 246, 0.18);}
.dd-ev .dd-src {font-style: italic; color: var(--body-text-color-subdued); font-size: 0.78rem; margin-top: 6px;}
.dd-summary {font-size: 0.86rem; line-height: 1.55; padding: 10px 12px; border-radius: 8px;
             background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);}
.dd-banner {padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border-color-primary);
            background: var(--background-fill-secondary); font-size: 0.9rem;}
"""

KIND_STYLE = {"artifact": ("Artefact", "#e5484d"), "scope": ("Scope", "#f5a524"),
              "limitation": ("Limitation", "#8e8c99"), "authentic": ("Authentic cue", "#30a46c")}


def _rgb(c):
    return f"rgb{tuple(c)}"


def nice(label: str) -> str:
    return label.replace("_", " ")


def header() -> str:
    return ("<div class='dd-header'><h1>Explainable Deepfake Detector</h1>"
            "<p>EfficientNet-B0 detector · Grad-CAM++ evidence · explanation grounded in forensic literature</p></div>")


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


def placeholder(text: str) -> str:
    return f"<div class='dd-card dd-muted'>{escape(text)}</div>"


def verdict(res, prompt: str = "") -> str:
    if res is None or res.prob_fake is None:
        return placeholder(prompt or "Analysing…")
    cls = "dd-fake" if res.label == "fake" else "dd-real"
    color = "#e5484d" if res.label == "fake" else "#30a46c"
    note = ""
    if res.face_found is False:
        note = "<div class='dd-muted'>No face detected: the whole image was analysed.</div>"
    chips = f"<span class='dd-chip'>{res.confidence} confidence</span>"
    if res.cached:
        chips += "<span class='dd-chip'>cached result</span>"
    truth = ""
    if res.truth:
        t = res.truth
        right = (res.label == "fake") == (t["label"] == "fake")
        mark = ("<span class='dd-ok'>✓ detector correct</span>" if right
                else "<span class='dd-bad'>✕ detector wrong</span>")
        truth = (f"<div class='dd-truth'>Actual: <b>{t['label'].upper()}</b> · "
                 f"{escape(METHOD_NAMES.get(t['manipulation'], t['manipulation']))} "
                 f"<span class='dd-muted'>({escape(t['dataset'])})</span><br>{mark}</div>")
    return (f"<div class='dd-card'><div class='dd-title'>Verdict</div>"
            f"<div class='dd-verdict {cls}'>{res.label.upper()}</div>"
            f"<div class='dd-gauge'><span style='width:{res.prob_fake * 100:.1f}%;background:{color}'></span>"
            f"<i style='left:calc({res.threshold * 100:.1f}% - 1px)' title='decision threshold'></i></div>"
            f"<div>P(fake) = <b>{res.prob_fake:.3f}</b> <span class='dd-muted'>· threshold {res.threshold:.3f}</span></div>"
            f"<div style='margin-top:6px'>{chips}</div>{note}{truth}</div>")


def regions(res) -> str:
    if res is None or not res.top_regions:
        return placeholder("The regions the detector relied on most will appear here.")
    top = max(c for _, c, _ in res.top_regions) or 1
    rows = []
    for i, (name, conc, share) in enumerate(res.top_regions):
        c = _rgb(TOP_COLORS[i])
        boxed = "" if name != BACKGROUND else " <span class='dd-muted'>(outside the face boxes)</span>"
        rows.append(f"<div class='dd-region'><span class='dd-num' style='background:{c}'>{i + 1}</span><div>"
                    f"<b>{escape(name)}</b>{boxed} <span class='dd-muted'>· {conc:.1f}× the image's average heat · "
                    f"{share:.0%} of all heat</span><div class='dd-bar' style='width:{conc / top * 100:.0f}%;"
                    f"background:{c}'></div></div></div>")
    title = "Where the detector looked: top 3 regions"
    src = "" if res.region_source == "landmarks" else \
        "<div class='dd-muted'>Region boxes from a fixed grid (no facial landmarks found).</div>"
    return f"<div class='dd-card'><div class='dd-title'>{title}</div>{''.join(rows)}{src}</div>"


def summary(res) -> str:
    if res is None or not res.summary:
        return placeholder("The evidence summary sent to the LLM will appear here.")
    return ("<div class='dd-card'><div class='dd-title'>Evidence summary · the only input the LLM reads "
            f"(it never sees the image)</div><div class='dd-summary dd-mono'>{escape(res.summary)}</div></div>")


def explanation(res, waiting: bool = False) -> str:
    if res is None or (res.answer is None and not waiting and not res.cards):
        return placeholder("The grounded explanation will appear here.")
    model = MODELS.get(res.model, res.model)
    if res.answer is None:
        text = (f"Asking {model}…" if waiting else
                f"No stored answer from {model} for this image." if res.cached else "No explanation.")
        return placeholder(text)
    a = res.answer
    if not a.schema_ok:
        return placeholder(f"{model}'s reply could not be used: {(a.error or a.raw)[:200]}")
    hits, invented = set(res.check.get("hits", [])), set(res.check.get("invented", []))
    cites = "".join(f"<span class='dd-chip dd-mono' style='border-color:"
                    f"{'#3b82f6' if c in hits else '#e5484d'}'>{escape(c)}</span>" for c in dict.fromkeys(a.citations))
    artefact = ("<span class='dd-chip' style='border-color:#30a46c'>no artefact: judged authentic</span>"
                if a.artifact_type == "none" else
                f"<span class='dd-chip' style='border-color:#e5484d'>artefact: {escape(nice(a.artifact_type))}</span>")
    checks = [f"cites {len(hits)} of the {len(res.cards)} cards it was given"]
    if invented:
        checks.append(f"<span class='dd-bad'>{len(invented)} id(s) not given to it</span>")
    if res.check.get("artifact_backed") is not None:
        checks.append(("<span class='dd-ok'>✓</span> artefact backed by a cited card" if res.check["artifact_backed"]
                       else "<span class='dd-bad'>✕</span> artefact not backed by a cited card"))
    checks.append(("<span class='dd-ok'>✓</span> region in the heatmap's top 3" if res.check.get("region_in_top3")
                   else "<span class='dd-bad'>✕</span> region not in the heatmap's top 3"))
    latency = f" · {res.timings['Explain']:.1f} s" if "Explain" in res.timings else ""
    return (f"<div class='dd-card'><div class='dd-title'>Grounded explanation · {escape(model)}{latency}</div>"
            f"<span class='dd-chip'>region: {escape(a.claimed_region)}</span>{artefact}"
            f"<div class='dd-expl'>{escape(a.explanation)}</div>"
            f"<div class='dd-muted'>Sources cited: {cites or 'none'}</div>"
            f"<div class='dd-check'>{' · '.join(checks)}</div></div>")


def cards(res) -> str:
    if res is None or not res.cards:
        return placeholder("The evidence cards retrieved from the corpus will appear here.")
    by_id, cited = cards_by_id(), set(res.check.get("hits", [])) if res.answer is not None else set()
    out = []
    for i, cid in enumerate(res.cards, 1):
        d = by_id.get(cid)
        if d is None:
            continue
        kind, color = KIND_STYLE.get(d.kind, (d.kind, "#8e8c99"))
        about = f" · {escape(nice(d.artifact_type))}" if d.kind in ("artifact", "scope") else ""
        badge = ("<span class='dd-chip' style='border-color:#3b82f6;color:#3b82f6'>cited ✓</span>" if cid in cited
                 else "<span class='dd-chip'>offered</span>" if res.answer is not None else "")
        out.append(f"<div class='dd-ev{' cited' if cid in cited else ''}'>"
                   f"<div><span class='dd-chip' style='border-color:{color};color:{color}'>{kind}</span>{badge}</div>"
                   f"<div style='margin:6px 0 4px'><b class='dd-mono'>{i}. {escape(cid)}</b>{about}</div>"
                   f"<div>{escape(d.text)}</div><div class='dd-src'>{escape(d.citation)}</div></div>")
    n_cited = len(cited & set(res.cards))
    title = (f"Evidence cards retrieved from the corpus · {len(res.cards)} offered"
             + (f", {n_cited} cited" if res.answer is not None else ""))
    return f"<div class='dd-card'><div class='dd-title'>{title}</div><div class='dd-cards'>{''.join(out)}</div></div>"
