#!/usr/bin/env python3
"""Create a single-brain NWCF exemplar overlay for Canva/paper drafting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from stew_nwcf import compute_clique_cohesion_distance
from stew_original_topology_illustrative_plots import nwcf_representative


PROJECT_ROOT = Path(__file__).resolve().parent
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# Projected positions on the supplied lateral brain drawing. These are not 3-D
# stereotactic coordinates; they are a consistent paper-figure projection.
LATERAL_POS = {
    "AF3": (0.17, 0.39),
    "AF4": (0.33, 0.27),
    "F7": (0.11, 0.50),
    "F3": (0.33, 0.40),
    "FC5": (0.27, 0.55),
    "T7": (0.18, 0.64),
    "P7": (0.43, 0.70),
    "O1": (0.76, 0.56),
    "O2": (0.86, 0.40),
    "P8": (0.65, 0.45),
    "T8": (0.53, 0.52),
    "FC6": (0.47, 0.42),
    "F4": (0.50, 0.29),
    "F8": (0.61, 0.25),
}

DORSAL_POS = {
    "AF3": (0.39, 0.12),
    "AF4": (0.61, 0.12),
    "F7": (0.20, 0.27),
    "F3": (0.39, 0.28),
    "FC5": (0.31, 0.42),
    "T7": (0.13, 0.50),
    "P7": (0.27, 0.70),
    "O1": (0.40, 0.87),
    "O2": (0.60, 0.87),
    "P8": (0.73, 0.70),
    "T8": (0.87, 0.50),
    "FC6": (0.69, 0.42),
    "F4": (0.61, 0.28),
    "F8": (0.80, 0.27),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def line_with_label(draw: ImageDraw.ImageDraw, xy0, xy1, text: str, fill, width: int = 3) -> None:
    draw.line([xy0, xy1], fill=fill, width=width)
    mx = int((xy0[0] + xy1[0]) / 2)
    my = int((xy0[1] + xy1[1]) / 2)
    draw.text((mx + 12, my - 10), text, fill=fill, font=font(24))


def draw_callout(draw: ImageDraw.ImageDraw, box, title: str, body: str, color) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=(255, 255, 255, 235), outline=color, width=3)
    draw.text((x0 + 18, y0 + 14), title, fill=color, font=font(29, bold=True))
    yy = y0 + 54
    for line in body.split("\n"):
        draw.text((x0 + 18, yy), line, fill=(45, 45, 45, 255), font=font(24))
        yy += 30


def curved_edge(draw: ImageDraw.ImageDraw, p0, p1, fill, width: int, bend: float = 0.18) -> None:
    x0, y0 = p0
    x1, y1 = p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx = mx - dy * bend
    cy = my + dx * bend
    pts = []
    for t in np.linspace(0, 1, 36):
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
        pts.append((int(x), int(y)))
    draw.line(pts, fill=fill, width=width, joint="curve")


def dashed_line(draw: ImageDraw.ImageDraw, p0, p1, fill, width: int = 2, dash: int = 16, gap: int = 10) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dist = float(np.hypot(x1 - x0, y1 - y0))
    if dist <= 0:
        return
    ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
    t = 0.0
    while t < dist:
        t2 = min(t + dash, dist)
        draw.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * t2, y0 + uy * t2)], fill=fill, width=width)
        t += dash + gap


def draw_region(draw: ImageDraw.ImageDraw, positions: dict[str, tuple[int, int]], nodes: list[str], label: str, fill, outline, label_xy) -> None:
    poly = [positions[n] for n in nodes if n in positions]
    if len(poly) < 3:
        return
    draw.polygon(poly, fill=fill)
    draw.line(poly + [poly[0]], fill=outline, width=2)
    draw.text(label_xy, label, fill=outline[:3] + (210,), font=font(20, bold=True))


def draw_whole_brain_parcels(draw: ImageDraw.ImageDraw, brain_xy: tuple[int, int], brain_w: int, brain_h: int) -> None:
    bx, by = brain_xy

    def pt(x: float, y: float) -> tuple[int, int]:
        return int(bx + x * brain_w), int(by + y * brain_h)

    parcels = [
        ("frontal control", [pt(0.12, 0.02), pt(0.88, 0.02), pt(0.86, 0.30), pt(0.66, 0.34), pt(0.50, 0.29), pt(0.34, 0.34), pt(0.14, 0.30)], (255, 188, 132, 62), (188, 92, 42, 155), pt(0.20, 0.18)),
        ("fronto-central scaffold", [pt(0.22, 0.29), pt(0.78, 0.29), pt(0.74, 0.53), pt(0.58, 0.58), pt(0.42, 0.58), pt(0.26, 0.53)], (244, 142, 174, 55), (170, 62, 102, 150), pt(0.36, 0.47)),
        ("left bridge", [pt(0.07, 0.30), pt(0.30, 0.40), pt(0.36, 0.74), pt(0.25, 0.86), pt(0.08, 0.64)], (183, 226, 170, 55), (68, 135, 82, 145), pt(0.09, 0.57)),
        ("right bridge", [pt(0.93, 0.30), pt(0.70, 0.40), pt(0.64, 0.74), pt(0.75, 0.86), pt(0.92, 0.64)], (183, 226, 170, 55), (68, 135, 82, 145), pt(0.75, 0.57)),
        ("posterior closure", [pt(0.25, 0.70), pt(0.75, 0.70), pt(0.82, 0.92), pt(0.67, 0.99), pt(0.33, 0.99), pt(0.18, 0.92)], (132, 202, 230, 58), (38, 118, 154, 148), pt(0.40, 0.86)),
        ("temporal belt", [pt(0.18, 0.50), pt(0.82, 0.50), pt(0.75, 0.72), pt(0.25, 0.72)], (205, 180, 226, 46), (112, 78, 150, 135), pt(0.42, 0.66)),
    ]
    for label, poly, fill, outline, label_xy in parcels:
        draw.polygon(poly, fill=fill)
        draw.line(poly + [poly[0]], fill=outline, width=2)
        draw.text(label_xy, label, fill=outline[:3] + (210,), font=font(20, bold=True))


def draw_support_shape(draw: ImageDraw.ImageDraw, positions: dict[str, tuple[int, int]], nodes: list[str], fill, outline, width: int = 4) -> None:
    poly = [positions[n] for n in nodes if n in positions]
    if len(poly) < 3:
        return
    draw.polygon(poly, fill=fill)
    draw.line(poly + [poly[0]], fill=outline, width=width)
    if len(poly) == 4:
        draw.line([poly[0], poly[2]], fill=outline, width=max(2, width - 1))
        draw.line([poly[1], poly[3]], fill=outline, width=max(2, width - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-image", required=True)
    parser.add_argument("--nwcf-config", default="84corr")
    parser.add_argument("--view", choices=["lateral", "dorsal"], default="dorsal")
    parser.add_argument("--output-dir", default="stew_original_topology_illustrative_figures")
    args = parser.parse_args()

    outdir = PROJECT_ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    dist, labels, tag = nwcf_representative(args.nwcf_config, "hi", "whole_head")
    d_hat, diag = compute_clique_cohesion_distance(dist, return_diagnostics=True, max_clique_order=4)
    sim = 1.0 - dist
    raw = dist[np.triu_indices_from(dist, k=1)]
    theta = float(np.percentile(raw, 18))
    shrink = dist - d_hat

    label_to_i = {lab: i for i, lab in enumerate(labels)}
    base_edges = []
    nwcf_edges = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if dist[i, j] <= theta:
                base_edges.append((i, j, sim[i, j]))
            if d_hat[i, j] <= theta and dist[i, j] > theta:
                nwcf_edges.append((i, j, shrink[i, j]))
    base_edges = sorted(base_edges, key=lambda e: e[2], reverse=True)[:24]
    nwcf_edges = sorted(nwcf_edges, key=lambda e: e[2], reverse=True)[:8]

    canvas = Image.new("RGBA", (1900, 1220), (255, 255, 255, 255))
    base = Image.open(args.brain_image).convert("RGBA")
    if args.view == "dorsal":
        crop = base.crop((48, 25, 282, 344))
    else:
        crop = base.crop((68, 38, 592, 425))
    crop = ImageEnhance.Contrast(crop).enhance(1.25)
    crop = ImageEnhance.Brightness(crop).enhance(1.05)
    if args.view == "dorsal":
        brain_w, brain_h = 900, 955
        brain_xy = (245, 158)
    else:
        brain_w, brain_h = 1210, 890
        brain_xy = (110, 170)
    crop = crop.resize((brain_w, brain_h), Image.Resampling.LANCZOS)
    canvas.alpha_composite(crop, brain_xy)
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.text((70, 42), "NWCF localizes a mesoscale workload scaffold in one STEW window", fill=(0, 0, 0, 255), font=font(42, bold=True))
    draw.text((72, 96), "Projected EEG channels; red arcs mark edges whose filtration entry is advanced by clique support.", fill=(75, 98, 110, 255), font=font(27))

    bx, by = brain_xy
    pos_template = DORSAL_POS if args.view == "dorsal" else LATERAL_POS
    positions = {ch: (int(bx + x * brain_w), int(by + y * brain_h)) for ch, (x, y) in pos_template.items()}

    if args.view == "dorsal":
        draw_whole_brain_parcels(draw, brain_xy, brain_w, brain_h)

    # Mesoscale support polygons chosen from high-shrinkage NWCF motifs.
    support_polys = (
        []
        if args.view == "lateral"
        else [
            (["F7", "F3", "FC5"], (52, 183, 204, 46), (20, 142, 165, 200), "triangle"),
            (["F8", "F4", "FC6"], (52, 183, 204, 46), (20, 142, 165, 200), "triangle"),
            (["F3", "F4", "FC6", "FC5"], (145, 80, 170, 36), (115, 55, 145, 190), "4-clique"),
            (["P7", "O1", "O2", "P8"], (145, 80, 170, 24), (115, 55, 145, 155), "4-clique"),
        ]
    )
    for nodes, fill, outline, _kind in support_polys:
        draw_support_shape(draw, positions, nodes, fill, outline, width=4)

    # Classical FC background: sparse context only.
    for i, j, _ in base_edges[:10]:
        ch0, ch1 = labels[i], labels[j]
        if ch0 in positions and ch1 in positions:
            curved_edge(draw, positions[ch0], positions[ch1], (82, 82, 82, 105), 3, bend=0.08)

    # NWCF-advanced edges in the foreground.
    for k, (i, j, _) in enumerate(nwcf_edges):
        ch0, ch1 = labels[i], labels[j]
        if ch0 in positions and ch1 in positions:
            curved_edge(draw, positions[ch0], positions[ch1], (208, 55, 45, 238), 8, bend=0.14 + 0.025 * (k % 3))

    node_counts = {ch: 0 for ch in labels}
    for i, j, _ in nwcf_edges:
        node_counts[labels[i]] = node_counts.get(labels[i], 0) + 1
        node_counts[labels[j]] = node_counts.get(labels[j], 0) + 1
    nwcf_nodes = {ch for ch, count in node_counts.items() if count > 0}
    for ch in nwcf_nodes:
        if ch in positions:
            x, y = positions[ch]
            radius = 20 + 5 * min(node_counts.get(ch, 1), 3)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 204, 92, 34), outline=(208, 55, 45, 155), width=3)

    # Channel markers and labels.
    for ch in CHANNELS:
        x, y = positions[ch]
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(255, 255, 255, 245), outline=(20, 20, 20, 255), width=3)
        dx = -38 if ch.endswith("3") or ch in {"F7", "FC5", "T7", "P7", "O1"} else 18
        dy = -36 if ch.startswith("A") or ch.startswith("F") else 18
        draw.text((x + dx, y + dy), ch, fill=(20, 20, 20, 255), font=font(21, bold=True))

    # Interpretation callouts: this figure's unique claim is spatial localization.
    draw_callout(
        draw,
        (1245, 195, 1795, 390),
        "Mesoscale scaffold",
        "NWCF localizes a supported\nfronto-central scaffold embedded\nin the broader FC context.",
        (35, 120, 140, 255),
    )
    draw_callout(
        draw,
        (1245, 455, 1795, 622),
        "NWCF-advanced scaffold",
        "Red arcs and gold nodes mark\nthe projected workload scaffold\nselected by clique support.",
        (180, 48, 40, 255),
    )
    dashed_line(draw, (1240, 292), positions["FC5"], (35, 120, 140, 180), 3)
    dashed_line(draw, (1240, 530), positions["T8"], (180, 48, 40, 190), 3)

    # Legend.
    lx, ly = 1238, 710
    draw.rounded_rectangle((lx, ly, lx + 565, ly + 290), radius=13, fill=(255, 255, 255, 238), outline=(180, 180, 180, 255), width=2)
    draw.text((lx + 24, ly + 20), "Visual encoding", fill=(0, 0, 0, 255), font=font(27, bold=True))
    curved_edge(draw, (lx + 34, ly + 76), (lx + 150, ly + 76), (92, 92, 92, 92), 3, bend=0.0)
    draw.text((lx + 174, ly + 61), "classical FC context", fill=(55, 55, 55, 255), font=font(24))
    curved_edge(draw, (lx + 34, ly + 125), (lx + 150, ly + 125), (208, 55, 45, 238), 8, bend=0.0)
    draw.text((lx + 174, ly + 108), "NWCF-advanced edge", fill=(150, 45, 38, 255), font=font(24))
    draw.polygon([(lx + 44, ly + 178), (lx + 92, ly + 150), (lx + 140, ly + 178)], fill=(52, 183, 204, 46), outline=(20, 142, 165, 200))
    draw.line([(lx + 44, ly + 178), (lx + 92, ly + 150), (lx + 140, ly + 178), (lx + 44, ly + 178)], fill=(20, 142, 165, 200), width=3)
    draw.text((lx + 174, ly + 154), "triangle support motif", fill=(20, 120, 145, 255), font=font(24))
    draw.rectangle((lx + 44, ly + 222, lx + 140, ly + 262), fill=(145, 80, 170, 36), outline=(115, 55, 145, 190), width=3)
    draw.line([(lx + 44, ly + 222), (lx + 140, ly + 262)], fill=(115, 55, 145, 190), width=2)
    draw.line([(lx + 44, ly + 262), (lx + 140, ly + 222)], fill=(115, 55, 145, 190), width=2)
    draw.text((lx + 174, ly + 224), "4-clique support motif", fill=(100, 45, 130, 255), font=font(24))

    gx, gy = 1248, 1040
    draw.rounded_rectangle((gx, gy, gx + 555, gy + 120), radius=13, fill=(255, 255, 255, 238), outline=(190, 190, 190, 255), width=2)
    draw.text((gx + 22, gy + 18), "Entry-time interpretation", fill=(0, 0, 0, 255), font=font(25, bold=True))
    draw.line((gx + 32, gy + 78, gx + 195, gy + 78), fill=(80, 80, 80, 150), width=3)
    draw.line((gx + 32, gy + 78, gx + 110, gy + 78), fill=(208, 55, 45, 235), width=7)
    draw.polygon([(gx + 110, gy + 78), (gx + 128, gy + 68), (gx + 128, gy + 88)], fill=(208, 55, 45, 235))
    draw.text((gx + 218, gy + 62), "clique support pulls edge entry earlier", fill=(75, 98, 110, 255), font=font(22))

    draw.text((245, 1128), "Pastel parcels segment the projected EEG network; triangle/4-clique motifs show local support behind NWCF-advanced red arcs.", fill=(75, 98, 110, 255), font=font(22))

    out = outdir / f"NWCFWorking_02A_BrainExemplarOverlay_{args.view}_{args.nwcf_config}.png"
    canvas.convert("RGB").save(out, quality=95)
    coords_out = outdir / f"NWCFWorking_02A_BrainExemplarOverlay_{args.view}_{args.nwcf_config}_coordinates.csv"
    with coords_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["type", "source", "target", "x", "y", "value"])
        for ch, (x, y) in positions.items():
            writer.writerow(["channel", ch, "", x, y, ""])
        for i, j, val in base_edges:
            writer.writerow(["classical_fc_edge", labels[i], labels[j], "", "", f"{val:.6f}"])
        for i, j, val in nwcf_edges:
            writer.writerow(["nwcf_advanced_edge", labels[i], labels[j], "", "", f"{val:.6f}"])
        for nodes, _, _, kind in support_polys:
            writer.writerow([kind, "-".join(nodes), "", "", "", ""])
    print(out)
    print(coords_out)


if __name__ == "__main__":
    main()
