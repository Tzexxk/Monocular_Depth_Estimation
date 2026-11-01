"""Build matching Mermaid sources and SVG figures using only the standard library."""

import argparse
import html
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    {
        "name": "overall_pipeline", "size": (1200, 1140),
        "title": "Overall monocular depth evaluation pipeline",
        "nodes": [
            ("A", 320, 110, 560, 90, "Prescribed C3VD sequences",
             ["Development: trans_t1_a (61 frames)", "Test: trans_t2_b (103 frames)"]),
            ("B", 320, 250, 560, 90, "Data preparation and audit",
             ["ZIP CRC, frame pairing, uint16 and geometry checks"]),
            ("C", 320, 390, 560, 100, "Shared C3VDDataset + DataLoader",
             ["Original geometry; RGB, depth_mm and valid_mask", "CPU loading: batch size 1, no shuffle"]),
            ("D", 60, 580, 500, 105, "Constant-median-depth baseline",
             ["One median fitted on development GT", "Fixed value for all test frames"]),
            ("E", 640, 580, 500, 105, "Depth Anything V2 + residual refinement",
             ["Frozen dev inverse mapping; optional gated CNN", "Official full-grid output: align_corners=True"]),
            ("F", 320, 790, 560, 110, "Common evaluation",
             ["Original GT mask; identical 103 test frames", "AbsRel | RMSE (mm) | delta < 1.25"]),
            ("G", 320, 990, 560, 90, "Comparison and report",
             ["Per-frame means, visual cases and limitations"]),
        ],
        "edges": [
            ("A", "B", [(600, 200), (600, 250)]),
            ("B", "C", [(600, 340), (600, 390)]),
            ("C", "D", [(600, 490), (600, 535), (310, 535), (310, 580)]),
            ("C", "E", [(600, 490), (600, 535), (890, 535), (890, 580)]),
            ("D", "F", [(310, 685), (310, 740), (460, 740), (460, 790)]),
            ("E", "F", [(890, 685), (890, 740), (740, 740), (740, 790)]),
            ("F", "G", [(600, 900), (600, 990)]),
        ],
    },
    {
        "name": "data_workflow", "size": (1200, 1510),
        "title": "Shared data preparation and loading workflow",
        "nodes": [
            ("A", 320, 110, 560, 90, "Official registered-video ZIP archives",
             ["trans_t1_a.zip | trans_t2_b.zip"]),
            ("B", 320, 250, 560, 100, "prepare_data.py",
             ["Archive, pair-count and image checks", "Audit report and development previews"]),
            ("C", 60, 430, 500, 100, "Unmodified raw frames",
             ["RGB PNG + uint16 depth TIFF", "data/raw/SEQUENCE/FRAME_ID_*"]),
            ("D", 640, 430, 500, 100, "Portable manifests",
             ["dev.csv: 61 | test.csv: 103", "Relative paths and preserved frame IDs"]),
            ("E", 320, 620, 560, 90, "C3VDDataset(data_root, split)",
             ["Resolve manifest, sequence and paired file paths"]),
            ("F", 320, 810, 560, 110, "Shared read_pair / decode_depth",
             ["depth_mm = raw_uint16 / 65535 * 100", "valid_mask = (raw > 0) & (raw < 65535)"]),
            ("G", 320, 1010, 560, 100, "Original-size sample dictionary",
             ["RGB float32 [3,H,W] in [0,1]", "Depth float32 [H,W] in mm; mask bool [H,W]"]),
            ("H", 320, 1190, 560, 90, "CPU DataLoader",
             ["batch_size=1 | num_workers=0 | shuffle=False"]),
            ("I", 320, 1370, 560, 90, "Inputs for depth estimation",
             ["Model-specific processing stays outside Dataset"]),
        ],
        "edges": [
            ("A", "B", [(600, 200), (600, 250)]),
            ("B", "C", [(600, 350), (600, 390), (310, 390), (310, 430)]),
            ("B", "D", [(600, 350), (600, 390), (890, 390), (890, 430)]),
            ("C", "E", [(310, 530), (310, 575), (460, 575), (460, 620)]),
            ("D", "E", [(890, 530), (890, 575), (740, 575), (740, 620)]),
            ("E", "F", [(600, 710), (600, 810)]),
            ("F", "G", [(600, 920), (600, 1010)]),
            ("G", "H", [(600, 1110), (600, 1190)]),
            ("H", "I", [(600, 1280), (600, 1370)]),
        ],
    },
    {
        "name": "constant_median_method", "size": (1200, 1300),
        "title": "Constant-median-depth baseline",
        "nodes": [
            ("A", 60, 110, 500, 90, "Development ground truth",
             ["trans_t1_a | 61 frames"]),
            ("B", 60, 270, 500, 110, "Select valid depth pixels",
             ["Decode uint16 GT to millimeters", "Keep 0 < raw < 65535"]),
            ("C", 60, 450, 500, 110, "Fit one global median",
             ["Pool valid pixels across all 61 frames", "c = median(all valid development depths)"]),
            ("D", 60, 630, 500, 110, "Freeze scalar c",
             ["One constant for the whole test sequence", "No fitting on test ground truth"]),
            ("E", 640, 450, 500, 110, "Test RGB frames",
             ["trans_t2_b | 103 frames", "Read H and W; RGB content is unused"]),
            ("F", 640, 630, 500, 110, "Broadcast constant depth",
             ["pred_i[u] = c for every pixel", "[H,W] float32 | millimeters"]),
            ("G", 60, 850, 500, 110, "Test ground truth and valid mask",
             ["trans_t2_b | same 103 frames", "Original GT grid: 1350 x 1080"]),
            ("H", 640, 850, 500, 110, "Per-frame evaluation",
             ["Original GT mask; finite positive predictions", "AbsRel | RMSE (mm) | delta < 1.25"]),
            ("I", 320, 1130, 560, 110, "Average per-frame scores",
             ["Arithmetic mean over 103 test frames", "Save metrics, constant value and configuration"]),
        ],
        "edges": [
            ("A", "B", [(310, 200), (310, 270)]),
            ("B", "C", [(310, 380), (310, 450)]),
            ("C", "D", [(310, 560), (310, 630)]),
            ("D", "F", [(560, 685), (640, 685)]),
            ("E", "F", [(890, 560), (890, 630)]),
            ("F", "H", [(890, 740), (890, 850)]),
            ("G", "H", [(560, 905), (640, 905)]),
            ("H", "I", [(890, 960), (890, 1050), (600, 1050), (600, 1130)]),
        ],
    },
]

FIGURES.append({
    "name": "pretrained_refinement_method", "size": (1200, 1740),
    "title": "Depth Anything V2 and gated residual refinement",
    "nodes": [
        ("A", 220, 110, 760, 100, "Shared original RGB input",
         ["RGB float32 [3,1080,1350] in [0,1]"]),
        ("B", 220, 310, 760, 110, "Frozen Depth Anything V2 Relative Small",
         ["ViT-S; 518 lower bound; ImageNet normalization",
          "Full-grid relative output: bilinear, align_corners=True"]),
        ("C", 220, 520, 760, 100, "Frozen development-set inverse calibration",
         ["D0 = 1 / (a * relative + b), in millimeters"]),
        ("D", 220, 720, 760, 110, "RGB and depth feature encoders",
         ["RGB features: 16 channels; depth features: 16 channels",
          "Depth input D0 / 100; HSV candidate mask: 1 channel"]),
        ("E", 220, 930, 760, 110, "Highlight-aware gated fusion",
         ["G = sigmoid(gate([RGB features, depth features, mask]))",
          "Fused features = depth features + G * RGB features"]),
        ("F", 220, 1140, 760, 110, "Multi-scale encoder and decoder",
         ["Encoder 32 / 64; bottleneck 128; decoder 64 / 32",
          "Skip connections; bilinear upsampling; residual output"]),
        ("G", 220, 1350, 760, 110, "Residual depth prediction",
         ["Normalized prediction = D0 / 100 + learned residual",
          "Multiply by 100; original grid float32 millimeters"]),
        ("H", 220, 1560, 760, 110, "Shared strict evaluation",
         ["Original depth_mm and GT mask; no prediction clipping",
          "C0 uses the same architecture with a zero highlight mask"]),
    ],
    "edges": [
        ("A", "B", [(600, 210), (600, 310)]),
        ("B", "C", [(600, 420), (600, 520)]),
        ("C", "D", [(600, 620), (600, 720)]),
        ("A", "D", [(980, 160), (1080, 160), (1080, 775), (980, 775)]),
        ("D", "E", [(600, 830), (600, 930)]),
        ("E", "F", [(600, 1040), (600, 1140)]),
        ("F", "G", [(600, 1250), (600, 1350)]),
        ("C", "G", [(220, 570), (120, 570), (120, 1405), (220, 1405)]),
        ("G", "H", [(600, 1460), (600, 1560)]),
    ],
})



def mermaid_source(figure):
    lines = ["flowchart TD"]
    for node, _, _, _, _, title, body in figure["nodes"]:
        label = "<br/>".join(html.escape(text, quote=True) for text in [title, *body])
        lines.append(f'    {node}["{label}"]')
    for source, target, _ in figure["edges"]:
        lines.append(f"    {source} --> {target}")
    lines.append("    classDef default fill:#eff6ff,stroke:#2563eb,stroke-width:2px;")
    return "\n".join(lines) + "\n"


def svg_source(figure):
    width, height = figure["size"]
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": f"0 0 {width} {height}",
        "width": str(width), "height": str(height), "role": "img",
        "aria-labelledby": "figure-title figure-description",
        "font-family": "Arial, Helvetica, sans-serif",
    })
    ET.SubElement(root, "title", {"id": "figure-title"}).text = figure["title"]
    ET.SubElement(root, "desc", {"id": "figure-description"}).text = (
        "Method stages and their data flow are connected by arrows."
    )
    ET.SubElement(root, "rect", {"width": str(width), "height": str(height), "fill": "white"})
    definitions = ET.SubElement(root, "defs")
    marker = ET.SubElement(definitions, "marker", {
        "id": "arrow", "viewBox": "0 0 10 10", "refX": "9", "refY": "5",
        "markerWidth": "8", "markerHeight": "8", "orient": "auto",
    })
    ET.SubElement(marker, "path", {"d": "M 0 0 L 10 5 L 0 10 z", "fill": "#475569"})
    ET.SubElement(root, "text", {
        "x": str(width // 2), "y": "58", "text-anchor": "middle", "fill": "#0f172a",
        "font-size": "30", "font-weight": "bold",
    }).text = figure["title"]
    for _, _, points in figure["edges"]:
        path = "M " + " L ".join(f"{x} {y}" for x, y in points)
        ET.SubElement(root, "path", {
            "d": path, "fill": "none", "stroke": "#475569", "stroke-width": "2.5",
            "marker-end": "url(#arrow)", "stroke-linejoin": "round",
        })
    for node, x, y, box_width, box_height, title, body in figure["nodes"]:
        group = ET.SubElement(root, "g", {"id": f"node-{node}"})
        attributes = {
            "x": str(x), "y": str(y), "width": str(box_width), "height": str(box_height),
            "rx": "12", "fill": "#eff6ff", "stroke": "#2563eb", "stroke-width": "2",
        }
        ET.SubElement(group, "rect", attributes)
        first_y = y + box_height / 2 - len(body) * 12 + 8
        for index, text in enumerate([title, *body]):
            ET.SubElement(group, "text", {
                "x": str(x + box_width / 2), "y": str(first_y + index * 24),
                "text-anchor": "middle", "font-size": "22" if index == 0 else "18",
                "font-weight": "bold" if index == 0 else "normal", "fill": "#0f172a",
            }).text = text
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for figure in FIGURES:
        ids = {node[0] for node in figure["nodes"]}
        assert len(ids) == len(figure["nodes"])
        assert all(source in ids and target in ids for source, target, _ in figure["edges"])
        for extension, content in [("mmd", mermaid_source(figure)), ("svg", svg_source(figure))]:
            path = args.output_dir / f"{figure['name']}.{extension}"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            print(path)


if __name__ == "__main__":
    main()
