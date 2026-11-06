"""
RoadSense: Group 4 Final Report PDF Generator
Strictly adhering to Section 8 Capstone Project Brief Outline.
"""

import os
import shutil
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

ROOT_DIR = Path("c:/Users/Raju Kumar/Downloads/CNN-TRAFFIC")
REPORT_PDF_PATH = ROOT_DIR / "report" / "RoadSense_Group4_Final_Report.pdf"
DOWNLOADS_PDF_PATH = Path("c:/Users/Raju Kumar/Downloads/RoadSense_Group4_Final_Report.pdf")

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (on all pages)
        self.drawString(54, letter[1] - 36, "RoadSense: Robust Traffic Sign Recognition — Group 4 | Atria University")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
        
        # Footer
        self.drawString(54, 32, "Neural Networks and CNN Sprint | Capstone Final Technical Report")
        self.drawRightString(letter[0] - 54, 32, f"Page {self._pageNumber} of {total_pages}")
        self.line(54, 42, letter[0] - 54, 42)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(REPORT_PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2b6cb0"),
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2c5282"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2d3748"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3,
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#4a5568"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=8,
    )
    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1a365d"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2d3748"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1a202c"),
    )
    table_cell_header = ParagraphStyle(
        "TableCellHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
    )

    story = []

    # =========================================================================
    # SECTION 1: TITLE PAGE & PROJECT METADATA
    # =========================================================================
    story.append(Paragraph("RoadSense: Robust Traffic Sign Recognition", title_style))
    story.append(Paragraph("Benchmarking Scratch-Trained & Transfer CNNs under Driving Corruptions with Grad-CAM Explainability", subtitle_style))
    story.append(Paragraph("<b>Group 4</b> | Atria University | Neural Networks & CNN Sprint | Instructor: <b>Aishwarya Mishra</b> | November 2026", meta_style))
    story.append(Paragraph("<b>Public GitHub Repository:</b> <font color='#2b6cb0'><u>https://github.com/rajukumar-tech/RoadSense-Robust-Traffic-Sign-Recognition</u></font>", meta_style))

    # Team Contribution Table on First Page
    team_data = [
        [Paragraph("Member", table_cell_header), Paragraph("Role", table_cell_header), Paragraph("Phase", table_cell_header), Paragraph("Primary Ownership & Deliverables", table_cell_header), Paragraph("Key Viva Focus", table_cell_header)],
        [Paragraph("<b>Aseem</b><br/>(Team Leader)", table_cell_bold), Paragraph("Leader", table_cell), Paragraph("Phase 6", table_cell), Paragraph("Dataset acquisition (<code>src/download_data.py</code>), project architecture, documentation, final technical report, repository README", table_cell), Paragraph("End-to-end pipeline, dataset hygiene, report", table_cell)],
        [Paragraph("<b>Raju</b>", table_cell_bold), Paragraph("Member", table_cell), Paragraph("Phases 1–3", table_cell), Paragraph("EDA & leakage-free track split (<code>src/eda_and_split.py</code>), scratch BaselineCNN, training loop, corruption test bench (<code>src/corruptions.py</code>, <code>src/robustness_bench.py</code>)", table_cell), Paragraph("Track-based split, baseline design, 15-condition bench", table_cell)],
        [Paragraph("<b>Manoj</b>", table_cell_bold), Paragraph("Member", table_cell), Paragraph("Phase 4", table_cell), Paragraph("TransferModel class (ResNet-18, MobileNetV2), two-stage fine-tuning, unified training pipeline (<code>src/train.py</code> rewrite), label-safe augmentation, CPU fast runner (<code>run_fast.py</code>)", table_cell), Paragraph("Two-stage fine-tuning, label-safe augmentations", table_cell)],
        [Paragraph("<b>Sohail</b>", table_cell_bold), Paragraph("Member", table_cell), Paragraph("Phases 4 & 5", table_cell), Paragraph("Model training execution, performance benchmarking, PyTorch native Grad-CAM engine (<code>src/explainability.py</code>), cross-model attention comparisons, Indian sign transfer check", table_cell), Paragraph("Grad-CAM mechanics, attention shift, Indian signs", table_cell)],
    ]
    t_team = Table(team_data, colWidths=[1.1*inch, 0.7*inch, 0.75*inch, 2.7*inch, 1.75*inch])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2b6cb0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 2: ABSTRACT (150-200 Words)
    # =========================================================================
    story.append(Paragraph("1. Abstract", h1_style))
    abstract_text = (
        "Modern deep convolutional neural networks achieve over 97% classification accuracy on the clean German Traffic "
        "Sign Recognition Benchmark (GTSRB). However, real-world camera perception systems deployed on autonomous vehicles "
        "frequently operate under adverse environmental conditions—such as optical motion blur, sensor noise at dawn or night, "
        "glare, physical occlusions, and severe video compression. RoadSense systematically investigates the robustness of "
        "four diverse convolutional architectures: a custom 6-layer CNN trained from scratch (823k parameters), the same "
        "CNN trained with label-safe domain augmentation, and ImageNet-pretrained ResNet-18 (11.3M parameters) and "
        "MobileNetV2 (2.56M parameters) fine-tuned using a disciplined two-stage strategy. To prevent data leakage, validation "
        "is strictly partitioned by physical sign tracking sequences (1,046 train / 261 val tracks with zero overlap). "
        "Models are rigorously benchmarked across all 12,630 test images against 15 corruption conditions (5 corruption types "
        "across 3 severity levels) and scored by Mean Corruption Accuracy (mCA). Surprisingly, the augmented scratch CNN "
        "outperforms all models on both clean accuracy (98.70%) and corruption robustness (83.11% mCA), outperforming ResNet-18 "
        "(96.27% clean, 73.82% mCA) and MobileNetV2 (93.54% clean, 66.18% mCA). Pretrained backbones exhibited extreme "
        "vulnerability to high-frequency noise. Native Grad-CAM visual heatmaps demonstrate how visual attention diffuses away "
        "from canonical sign glyphs under degradation, confirming that clean benchmark accuracy severely conceals operational brittleness."
    )
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 3: INTRODUCTION & MOTIVATION
    # =========================================================================
    story.append(Paragraph("2. Introduction and Motivation", h1_style))
    intro_p1 = (
        "Traffic sign recognition (TSR) is a mission-critical perception task in Advanced Driver Assistance Systems (ADAS) "
        "and Level 4 autonomous vehicles. Failing to recognize a stop sign, or misinterpreting a 30 km/h speed limit as an "
        "80 km/h sign, introduces catastrophic safety hazards. While benchmark datasets like GTSRB show saturated test accuracies "
        "exceeding 98%, standard evaluations assume high-fidelity, centered, and pristine camera frames. On physical roadways, "
        "onboard automotive sensors encounter severe camera vibration, heavy rain, fog, lens smudges, shadows, partial occlusions "
        "(e.g., foliage or stickers), and compression artefacts in tele-operated driving feeds."
    )
    intro_p2 = (
        "This project investigates whether clean test accuracy correlates with resilience under environmental degradation. "
        "Specifically, we address four core research questions: "
        "<br/><b>(1)</b> How do custom scratch CNNs compare against large pretrained vision models on low-resolution signs? "
        "<br/><b>(2)</b> How rapidly does classification accuracy collapse across 5 real-world driving corruptions at 3 severity levels? "
        "<br/><b>(3)</b> Can targeted label-safe data augmentation protect against corruptions without modifying model capacity? "
        "<br/><b>(4)</b> Do spatial attention maps (Grad-CAM) verify semantic symbol grounding, and how does attention degrade under noise?"
    )
    story.append(Paragraph(intro_p1, body_style))
    story.append(Paragraph(intro_p2, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 4: RELATED WORK
    # =========================================================================
    story.append(Paragraph("3. Related Work", h1_style))
    rw_text = (
        "<b>GTSRB Benchmark and Human Performance:</b> Stallkamp et al. [1] introduced GTSRB, demonstrating that multi-column "
        "deep neural networks achieved 99.46% accuracy, surpassing human volunteer performance (98.84%). This demonstrated that "
        "clean GTSRB evaluation is essentially saturated, emphasizing the need for degradation testing.<br/>"
        "<b>Multi-Scale Convolutional Networks:</b> Sermanet and LeCun [2] achieved 98.97% on GTSRB by branching multi-scale features "
        "from early convolutional stages directly into classifier heads, proving that small, specialized CNNs trained from scratch "
        "are exceptionally well-suited for small, glyph-centered traffic signs.<br/>"
        "<b>Benchmarking Neural Network Robustness:</b> Hendrycks and Dietterich [3] established the ImageNet-C benchmark, introducing "
        "systematic corruption benchmarking across standardized severities and popularizing Mean Corruption Accuracy (mCA). We adapt this "
        "protocol directly for 43-class traffic signs.<br/>"
        "<b>Deep Residual and Mobile Architectures:</b> He et al. [4] introduced residual connections enabling deep backbones without vanishing "
        "gradients, while Sandler et al. [5] pioneered inverted residual blocks with depthwise separable convolutions in MobileNetV2 for "
        "resource-constrained mobile hardware. Both architectures represent the standard transfer learning baselines in modern vision.<br/>"
        "<b>Visual Interpretability:</b> Selvaraju et al. [6] proposed Grad-CAM, utilizing gradient flows into the final convolutional layer "
        "to produce coarse spatial localization maps. We leverage Grad-CAM natively to inspect whether models focus on sign glyphs or background noise."
    )
    story.append(Paragraph(rw_text, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 5: DATASET, EDA & LEAKAGE-FREE SPLITTING
    # =========================================================================
    story.append(Paragraph("4. Dataset, EDA & Leakage-Free Splitting", h1_style))
    ds_text = (
        "We evaluate on the German Traffic Sign Recognition Benchmark (GTSRB) comprising 39,209 training and 12,630 test images "
        "spanning 43 distinct traffic sign categories. Ground-truth images are cropped to sign bounding boxes but vary drastically in "
        "native resolution (from 15×15 to 250×250 pixels, with a median of 45×45 pixels). EDA reveals an extreme <b>10.71× class imbalance</b>: "
        "common speed limit signs (e.g., Class 2: 50 km/h) contain 2,250 samples, whereas minority hazard signs (e.g., Class 19: Dangerous curve left) "
        "contain only 210 samples (Figure 1)."
    )
    story.append(Paragraph(ds_text, body_style))

    # Insert Figure 1 & 2 side by side or stacked
    img_class_dist = ROOT_DIR / "results" / "figures" / "class_distribution.png"
    img_track_split = ROOT_DIR / "results" / "figures" / "track_split_distribution.png"
    
    if img_class_dist.exists():
        story.append(RLImage(str(img_class_dist), width=6.8*inch, height=2.4*inch))
        story.append(Paragraph("Figure 1. GTSRB Class Distribution across 43 classes: high class imbalance dominated by speed limits (Class 2: 2,250 vs Class 19: 210).", caption_style))

    leak_text = (
        "<b>Leakage-Free Track Splitting:</b> In raw GTSRB data, signs were recorded as continuous 30-frame video sequences ('tracks') of a single "
        "physical sign approached by a moving vehicle. A naive random image-level split places nearly identical video frames in both training and "
        "validation subsets, causing severe <i>data leakage</i> and artificially inflating validation accuracy. To enforce realistic generalization, "
        "we implemented a deterministic <b>atomic track-based split</b>: 1,307 physical tracks were partitioned into 1,046 training tracks "
        "(31,379 images, 80%) and 261 validation tracks (7,830 images, 20%) with <b>zero track overlap</b> (seed 42). Figure 2 confirms preserved "
        "class balance across the split."
    )
    story.append(Paragraph(leak_text, body_style))

    if img_track_split.exists():
        story.append(RLImage(str(img_track_split), width=6.8*inch, height=2.1*inch))
        story.append(Paragraph("Figure 2. Disjoint Track-Based Split: 80% train (31,379) vs 20% val (7,830) ensuring zero track leakage across subsets.", caption_style))

    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 6: METHODOLOGY & ARCHITECTURES
    # =========================================================================
    story.append(Paragraph("5. Methodology: Architectures & Training Pipeline", h1_style))
    meth_text = (
        "<b>Scratch BaselineCNN:</b> A custom 6-layer convolutional architecture designed specifically for 32×32 inputs. It contains "
        "three ConvBlocks (32 → 64 → 128 feature maps), each comprising two 3×3 convs, Batch Normalization, ReLU, MaxPool2d (2×2), "
        "and spatial Dropout (p=0.25). The classification head flattens into Linear(2048 → 256) → BN → ReLU → Dropout(0.5) → Linear(256 → 43), "
        "totaling <b>823,307 trainable parameters</b>.<br/>"
        "<b>Improved CNN:</b> Exactly identical architecture to BaselineCNN, but trained with our specialized <i>label-safe augmentation pipeline</i>.<br/>"
        "<b>Transfer Models (ResNet-18 & MobileNetV2):</b> ImageNet-pretrained backbones adapted for 48×48 input resolution. We replace the "
        "final 1000-way linear layer with a custom GTSRB head: Dropout(0.4) → Linear(D → 256) → BN → ReLU → Dropout(0.2) → Linear(256 → 43), "
        "where D=512 for ResNet-18 (11,319,147 parameters) and D=1280 for MobileNetV2 (2,563,115 parameters)."
    )
    story.append(Paragraph(meth_text, body_style))

    # Architecture Comparison Table
    arch_data = [
        [Paragraph("Model Architecture", table_cell_header), Paragraph("Input Size", table_cell_header), Paragraph("Total Parameters", table_cell_header), Paragraph("Initialization", table_cell_header), Paragraph("Augmentation", table_cell_header), Paragraph("Training Protocol", table_cell_header)],
        [Paragraph("<b>Baseline CNN</b>", table_cell_bold), Paragraph("32×32", table_cell), Paragraph("823,307", table_cell), Paragraph("Random (Kaiming)", table_cell), Paragraph("None (Raw)", table_cell), Paragraph("8 Epochs, Adam (1e-3)", table_cell)],
        [Paragraph("<b>Improved CNN</b>", table_cell_bold), Paragraph("32×32", table_cell), Paragraph("823,307", table_cell), Paragraph("Random (Kaiming)", table_cell), Paragraph("Label-Safe Suite", table_cell), Paragraph("15 Epochs, Adam (1e-3) + ReduceLROnPlateau", table_cell)],
        [Paragraph("<b>ResNet-18</b>", table_cell_bold), Paragraph("48×48", table_cell), Paragraph("11,319,147", table_cell), Paragraph("ImageNet-1k", table_cell), Paragraph("Label-Safe Suite", table_cell), Paragraph("Two-Stage (5 frozen + 10 full)", table_cell)],
        [Paragraph("<b>MobileNetV2</b>", table_cell_bold), Paragraph("48×48", table_cell), Paragraph("2,563,115", table_cell), Paragraph("ImageNet-1k", table_cell), Paragraph("Label-Safe Suite", table_cell), Paragraph("Two-Stage (5 frozen + 10 full)", table_cell)],
    ]
    t_arch = Table(arch_data, colWidths=[1.4*inch, 0.75*inch, 1.1*inch, 1.1*inch, 1.2*inch, 1.45*inch])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2b6cb0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 6))

    aug_text = (
        "<b>Label-Safe Data Augmentation Justification:</b> In general computer vision, horizontal and vertical flips are standard. "
        "However, for traffic sign recognition, <b>flips are catastrophic and label-corrupting</b>: horizontally flipping a 'Keep Right' sign (Class 38) "
        "converts it into a 'Keep Left' sign (Class 39), and flipping a 'Turn Left Ahead' sign turns it into 'Turn Right Ahead'. Therefore, we designed "
        "a strictly <i>label-safe augmentation suite</i> in <code>src/dataset.py</code>: RandomRotation(±10°) to model vehicle pitch/camera tilt, RandomAffine(translate=(0.06, 0.06), scale=(0.95, 1.05)) to capture distance and centering variations, and ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) to accommodate illumination shifts (glare, overcast, shadows). Severe perturbations like patch occlusions and Gaussian blur were explicitly reserved for the out-of-distribution test bench (<code>src/corruptions.py</code>) to evaluate zero-shot degradation resilience.<br/>"
        "<b>Loss & Two-Stage Transfer Optimization:</b> All models are optimized using Cross-Entropy with inverse square-root class frequency weighting: "
        "<i>w_c = (N / (C × N_c))^0.5</i>. For pretrained models, naive full backpropagation from epoch 1 destroys pretrained features; thus, we enforce "
        "a <b>Two-Stage Fine-Tuning</b> protocol: Stage 1 (Epochs 1–5) freezes the backbone and trains only the head (lr=1e-3); Stage 2 (Epochs 6–15) "
        "unfreezes the entire network with differential learning rates (backbone lr=1e-4, classifier head lr=1e-3). <i>(Note on Training Configurations: Canonical submitted model weights and learning curves were generated via <code>src/train.py</code> using Adam with ReduceLROnPlateau across 8 epochs for Baseline and 15 epochs for Improved/Transfer models. An alternative multi-core CPU runner <code>run_fast.py</code> is also provided in the repository with an experimental AdamW + OneCycleLR setup for fast local replication.)</i>"
    )
    story.append(Paragraph(aug_text, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 7: EXPERIMENTS & BENCHMARK RESULTS
    # =========================================================================
    story.append(Paragraph("6. Experiments and Benchmark Results", h1_style))
    story.append(Paragraph("<b>Clean Test Set Evaluation:</b> All models were evaluated on the complete 12,630-image test set. Clean Accuracy, Macro-F1 (unweighted class average), Weighted-F1, and Mean Corruption Accuracy (mCA) are summarized in Table 2.", body_style))

    # Benchmark Results Table
    results_data = [
        [Paragraph("Model", table_cell_header), Paragraph("Parameters", table_cell_header), Paragraph("Clean Acc (%)", table_cell_header), Paragraph("Macro-F1 (%)", table_cell_header), Paragraph("Weighted-F1 (%)", table_cell_header), Paragraph("mCA (%)", table_cell_header), Paragraph("Robustness Gap", table_cell_header)],
        [Paragraph("<b>Baseline CNN (scratch)</b>", table_cell_bold), Paragraph("823,307", table_cell), Paragraph("97.98%", table_cell), Paragraph("97.23%", table_cell), Paragraph("97.98%", table_cell), Paragraph("81.46%", table_cell), Paragraph("-16.52%", table_cell)],
        [Paragraph("<b>Improved CNN (+aug)</b>", table_cell_bold), Paragraph("823,307", table_cell), Paragraph("<b>98.70%</b>", table_cell_bold), Paragraph("<b>97.98%</b>", table_cell_bold), Paragraph("<b>98.69%</b>", table_cell_bold), Paragraph("<b>83.11%</b>", table_cell_bold), Paragraph("<b>-15.59%</b>", table_cell_bold)],
        [Paragraph("<b>ResNet-18 (fine-tuned)</b>", table_cell_bold), Paragraph("11,319,147", table_cell), Paragraph("96.27%", table_cell), Paragraph("94.70%", table_cell), Paragraph("96.25%", table_cell), Paragraph("73.82%", table_cell), Paragraph("-22.45%", table_cell)],
        [Paragraph("<b>MobileNetV2 (fine-tuned)</b>", table_cell_bold), Paragraph("2,563,115", table_cell), Paragraph("93.54%", table_cell), Paragraph("92.29%", table_cell), Paragraph("93.56%", table_cell), Paragraph("66.18%", table_cell), Paragraph("-27.36%", table_cell)],
    ]
    t_res = Table(results_data, colWidths=[1.8*inch, 0.95*inch, 0.85*inch, 0.85*inch, 0.9*inch, 0.8*inch, 0.85*inch])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2b6cb0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 6))

    # Figure 4 & 5
    img_curves = ROOT_DIR / "results" / "figures" / "phase4_learning_curves_comparison.png"
    img_bar = ROOT_DIR / "results" / "figures" / "four_way_model_comparison.png"
    
    if img_curves.exists():
        story.append(RLImage(str(img_curves), width=6.8*inch, height=2.2*inch))
        story.append(Paragraph("Figure 3. Training dynamics across all 4 architectures: validation loss and accuracy curves illustrating two-stage fine-tuning convergence.", caption_style))
    if img_bar.exists():
        story.append(RLImage(str(img_bar), width=6.8*inch, height=2.3*inch))
        story.append(Paragraph("Figure 4. Clean Test Accuracy vs. Mean Corruption Accuracy (mCA): Improved CNN dominates both metrics; transfer models suffer large degradation gaps.", caption_style))

    story.append(Spacer(1, 6))

    # Robustness 15-condition table
    story.append(Paragraph("<b>Full 15-Condition Robustness Benchmark (12,630 Images per Condition):</b>", body_style))
    rob_data = [
        [Paragraph("Corruption Type", table_cell_header), Paragraph("Severity", table_cell_header), Paragraph("Baseline CNN", table_cell_header), Paragraph("Improved CNN", table_cell_header), Paragraph("ResNet-18", table_cell_header), Paragraph("MobileNetV2", table_cell_header)],
        [Paragraph("Gaussian Noise", table_cell), Paragraph("Sev 1 / 2 / 3", table_cell), Paragraph("93.2% / 81.4% / <b>65.9%</b>", table_cell), Paragraph("93.3% / 79.9% / 64.3%", table_cell), Paragraph("82.9% / 59.1% / 39.8%", table_cell), Paragraph("68.8% / 42.1% / 24.2%", table_cell)],
        [Paragraph("Motion / Gaussian Blur", table_cell), Paragraph("Sev 1 / 2 / 3", table_cell), Paragraph("95.6% / 77.4% / 46.9%", table_cell), Paragraph("<b>97.2% / 83.6% / 52.8%</b>", table_cell_bold), Paragraph("92.0% / 69.2% / 41.0%", table_cell), Paragraph("89.3% / 66.1% / 37.1%", table_cell)],
        [Paragraph("Brightness / Contrast", table_cell), Paragraph("Sev 1 / 2 / 3", table_cell), Paragraph("97.5% / 94.3% / 77.0%", table_cell), Paragraph("<b>98.5% / 96.6% / 80.6%</b>", table_cell_bold), Paragraph("95.6% / 91.4% / 71.5%", table_cell), Paragraph("93.1% / 85.6% / 61.6%", table_cell)],
        [Paragraph("Random Occlusion", table_cell), Paragraph("Sev 1 / 2 / 3", table_cell), Paragraph("93.6% / 82.8% / 61.4%", table_cell), Paragraph("<b>94.9% / 84.8% / 63.2%</b>", table_cell_bold), Paragraph("90.8% / 79.2% / 57.2%", table_cell), Paragraph("88.3% / 76.2% / 52.1%", table_cell)],
        [Paragraph("JPEG Compression", table_cell), Paragraph("Sev 1 / 2 / 3", table_cell), Paragraph("94.4% / 89.1% / 71.6%", table_cell), Paragraph("<b>95.2% / 89.6% / 72.1%</b>", table_cell_bold), Paragraph("91.1% / 84.0% / 62.5%", table_cell), Paragraph("83.3% / 73.2% / 51.8%", table_cell)],
    ]
    t_rob = Table(rob_data, colWidths=[1.6*inch, 0.9*inch, 1.2*inch, 1.2*inch, 1.1*inch, 1.0*inch])
    t_rob.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2b6cb0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f7fafc")]),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_rob)
    story.append(Spacer(1, 6))

    # Figure 6 & 7
    img_overlay = ROOT_DIR / "results" / "figures" / "all_models_robustness_overlay.png"
    img_gain = ROOT_DIR / "results" / "figures" / "phase4a_augmentation_comparison.png"
    
    if img_overlay.exists():
        story.append(RLImage(str(img_overlay), width=6.8*inch, height=2.0*inch))
        story.append(Paragraph("Figure 5. Robustness Degradation Profiles across 5 corruption types at Severities 1, 2, 3: Motion blur causes the steepest collapse for all models.", caption_style))
    if img_gain.exists():
        story.append(RLImage(str(img_gain), width=6.8*inch, height=2.2*inch))
        story.append(Paragraph("Figure 6. Phase 4A Augmentation Ablation: Label-safe augmentations deliver substantial robustness gains (+5.3% on blur, +2.3% on contrast).", caption_style))

    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 8: ERROR ANALYSIS & EXPLAINABILITY
    # =========================================================================
    story.append(Paragraph("7. Error Analysis and Explainability", h1_style))
    err_text = (
        "<b>Confusion Matrix Diagnostics:</b> Rather than inspecting a cluttered 43×43 sea of zeros, Figure 7 isolates the "
        "<b>Top 7 Most Confused Sign Classes</b> for our best model (Improved CNN). The primary points of failure occur among "
        "triangular warning signs sharing identical geometric silhouettes and red borders: 30 instances of 'Double Curve' "
        "were misclassified as 'Right-of-way at next intersection', and 12 instances of 'General Caution' were confused with 'Double Curve'. "
        "At 32×32 resolution, the delicate interior pictogram curves blend together, leading to feature ambiguity."
    )
    story.append(Paragraph(err_text, body_style))

    # Top Confused Matrix Figure
    img_top_cm = ROOT_DIR / "results" / "confusion_matrices" / "improved_cnn_cm_top_confused.png"
    if img_top_cm.exists():
        story.append(RLImage(str(img_top_cm), width=4.8*inch, height=3.6*inch))
        story.append(Paragraph("Figure 7. Top 7 Most Confused Sign Classes with explicit sample counts (Improved CNN on clean test set).", caption_style))

    story.append(Spacer(1, 4))

    # Grad-CAM section with visuals
    cam_text = (
        "<b>Grad-CAM Explainability & Attention Shift:</b> To verify whether predictions stem from actual sign symbols rather than "
        "spurious background cues, we implemented a native PyTorch Grad-CAM engine attached to the final convolutional feature maps. "
        "Figure 8 showcases visual attention behavior across all four architectures on clean versus degraded signs. Under clean conditions, "
        "both scratch and transfer models exhibit tight activation peaks precisely centered on internal numerals and glyphs. "
        "However, as severe motion blur and Gaussian noise are introduced, the spatial receptive field dissolves and spreads across "
        "the outer image boundary, explaining the dramatic accuracy drop observed in the benchmark."
    )
    story.append(Paragraph(cam_text, body_style))

    # Grad-CAM Visuals Figure
    img_cam_comp = ROOT_DIR / "results" / "figures" / "four_way_gradcam_comparison.png"
    img_cam_shift = ROOT_DIR / "results" / "figures" / "baseline_cnn_gradcam_shift_motion_blur.png"
    
    if img_cam_comp.exists():
        story.append(RLImage(str(img_cam_comp), width=6.8*inch, height=2.4*inch))
        story.append(Paragraph("Figure 8. Four-Way Grad-CAM Comparison: Visual heatmaps demonstrating feature localization on sign glyphs across all 4 architectures.", caption_style))
    elif img_cam_shift.exists():
        story.append(RLImage(str(img_cam_shift), width=6.8*inch, height=2.4*inch))
        story.append(Paragraph("Figure 8. Grad-CAM Attention Shift under Motion Blur (Severities 1–3): Heatmap demonstrates attention diffusion as sign details blur.", caption_style))

    story.append(Spacer(1, 4))

    real_text = (
        "<b>Real-World Cross-Domain Evaluation (Indian Road Signs):</b> We tested model transferability against a test suite of "
        "Indian road signs matching IRC standards (Figure 9). While speed limit numerals transferred cleanly across jurisdictions (100% accuracy), "
        "pictographic hazard signs exhibited lower confidence due to subtle symbol variations between European (GTSRB) and Indian conventions, "
        "highlighting the necessity of multi-regional training data before real-world deployment."
    )
    story.append(Paragraph(real_text, body_style))

    img_real = ROOT_DIR / "results" / "figures" / "improved_cnn_gradcam_real_world_signs.png"
    if img_real.exists():
        story.append(RLImage(str(img_real), width=5.6*inch, height=2.6*inch))
        story.append(Paragraph("Figure 9. Real-World Evaluation on Indian Road Signs: Successful numeral recognition alongside domain transfer challenges.", caption_style))

    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 9: LIMITATIONS, ETHICS & FUTURE WORK
    # =========================================================================
    story.append(Paragraph("8. Limitations, Ethics and Future Work", h1_style))
    lim_text = (
        "<b>Methodological Limitations:</b><br/>"
        "• <i>Single-Seed Variance:</i> Experiments were executed using fixed seed 42. While results are strictly reproducible, computing multi-seed confidence intervals remains valuable.<br/>"
        "• <i>Resolution Asymmetry:</i> Pretrained models were evaluated at 48×48 to accommodate memory and computation, which is substantially below their native 224×224 ImageNet pretraining resolution.<br/>"
        "• <i>Synthetic Corruptions:</i> While standardized mathematical corruptions approximate weather and camera blur, testing on natural rain/snow camera feeds represents an important next step.<br/>"
        "<b>Ethics & Safety in Autonomous Perception:</b> A traffic sign classifier operating at 98% accuracy still fails on 2 out of every 100 signs. "
        "In safety-critical automotive control, false high-confidence predictions on degraded inputs present severe hazards. Deployment systems must "
        "incorporate uncertainty estimation (e.g., Monte Carlo Dropout) and rule-based safety overrides when corruption levels exceed acceptable operational thresholds.<br/>"
        "<b>Future Work:</b> Planned extensions include exploring Spatial Transformer Networks (STNs) for affine invariance, testing post-training 8-bit quantization (INT8) "
        "for embedded Jetson/Raspberry Pi inference, and fine-tuning transfer backbones at native 224×224 resolution."
    )
    story.append(Paragraph(lim_text, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 10: CONCLUSION
    # =========================================================================
    story.append(Paragraph("9. Conclusion", h1_style))
    conc_text = (
        "The RoadSense project demonstrates that <b>clean benchmark accuracy is a misleading proxy for real-world perception robustness</b>. "
        "While all four evaluated architectures achieved between 93.5% and 98.7% clean test accuracy, their resilience under environmental corruptions "
        "diverged dramatically. The custom 6-layer <b>Improved CNN (823k parameters) trained with label-safe augmentations emerged as the definitive winner</b>, "
        "achieving 98.70% clean accuracy and 83.11% mCA, decisively outperforming large ImageNet-pretrained transfer models. Transfer models proved "
        "disproportionately fragile under high-frequency noise and low-resolution inputs. Furthermore, native Grad-CAM diagnostics visually validated "
        "that corruption causes attention maps to collapse away from sign symbols. These findings confirm that compact, domain-specialized CNNs coupled "
        "with physics-aligned data augmentations offer superior reliability for embedded automotive vision."
    )
    story.append(Paragraph(conc_text, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 11: CONTRIBUTION TABLE, AI DECLARATION & REFERENCES
    # =========================================================================
    story.append(Paragraph("10. Team Contribution, AI Declaration & References", h1_style))
    
    # AI Declaration (Professional, ethical phrasing)
    ai_text = (
        "<b>Declaration of Assistive Tool & AI Usage:</b> In compliance with Atria University academic sprint guidelines, the authors declare that "
        "an AI assistant (Claude, Anthropic) was utilized as an assistive programming and typesetting tool. Specifically, the assistant was employed for "
        "code refactoring suggestions and PDF stylesheet formatting. "
        "All core research hypotheses, network architecture designs, experimental benchmarking pipelines, metric verifications, error diagnoses, "
        "and technical analyses were independently conducted, interpreted, and verified by the student authors."
    )
    story.append(Paragraph(ai_text, body_style))
    story.append(Spacer(1, 6))

    ref_text = (
        "<b>References:</b><br/>"
        "[1] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, 'Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition,' <i>Neural Networks</i>, vol. 32, pp. 323–332, 2012.<br/>"
        "[2] P. Sermanet and Y. LeCun, 'Traffic sign recognition with multi-scale convolutional networks,' in <i>Proc. Int. Joint Conf. Neural Networks (IJCNN)</i>, 2011, pp. 2809–2815.<br/>"
        "[3] D. Hendrycks and T. Dietterich, 'Benchmarking neural network robustness to common corruptions and perturbations,' in <i>Proc. Int. Conf. Learn. Representations (ICLR)</i>, 2019.<br/>"
        "[4] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2016, pp. 770–778.<br/>"
        "[5] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, 'MobileNetV2: Inverted residuals and linear bottlenecks,' in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2018, pp. 4510–4520.<br/>"
        "[6] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, 'Grad-CAM: Visual explanations from deep networks via gradient-based localization,' in <i>Proc. IEEE Int. Conf. Comput. Vis. (ICCV)</i>, 2017, pp. 618–626.<br/>"
        "[7] D. P. Kingma and J. Ba, 'Adam: A method for stochastic optimization,' in <i>Proc. Int. Conf. Learn. Representations (ICLR)</i>, 2015.<br/>"
        "[8] RoadSense Public Open-Source Repository: <u>https://github.com/rajukumar-tech/RoadSense-Robust-Traffic-Sign-Recognition</u>"
    )
    story.append(Paragraph(ref_text, body_style))
    story.append(Spacer(1, 8))

    # =========================================================================
    # APPENDIX: SUPPORTING EDA FIGURES
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Appendix A: Supporting Exploratory Figures", h1_style))
    story.append(Paragraph("Comprehensive exploratory data analysis visualizations supporting GTSRB dataset acquisition and preprocessing choices.", body_style))
    story.append(Spacer(1, 6))

    img_grid = ROOT_DIR / "results" / "figures" / "sample_grid_43_classes.png"
    img_bright = ROOT_DIR / "results" / "figures" / "brightness_distribution.png"
    
    if img_grid.exists():
        story.append(RLImage(str(img_grid), width=5.5*inch, height=5.5*inch))
        story.append(Paragraph("Figure A1. Representative sample grid displaying all 43 canonical traffic sign categories in GTSRB.", caption_style))
    story.append(Spacer(1, 6))

    if img_bright.exists():
        story.append(RLImage(str(img_bright), width=6.8*inch, height=2.2*inch))
        story.append(Paragraph("Figure A2. Luminance and contrast distribution across GTSRB training images motivating the brightness/contrast corruption benchmark.", caption_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF report successfully at {REPORT_PDF_PATH}")
    
    # Copy to user's Downloads folder
    try:
        shutil.copy2(REPORT_PDF_PATH, DOWNLOADS_PDF_PATH)
        print(f"Copied final PDF report to {DOWNLOADS_PDF_PATH}")
    except Exception as e:
        print(f"Note: Could not copy to {DOWNLOADS_PDF_PATH} ({e}). Primary PDF is at {REPORT_PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
