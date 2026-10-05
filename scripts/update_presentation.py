from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "Copy of Presented by H M Akshay- 1RF23CS068 Hemanth Kumar K- "
    "1RF23CS072 Shreya Gopalakrishna- 1RF23CS159 Venkat Baba Yemineni- "
    "1RF23CS181.pptx"
)
OUTPUT = ROOT / "Updated_Longitudinal_Temporal_Disease_Progression.pptx"
ASSETS = ROOT / "docs" / "assets" / "results"

NAVY = RGBColor(25, 50, 77)
TEAL = RGBColor(11, 110, 117)
GREEN = RGBColor(45, 138, 100)
ORANGE = RGBColor(232, 135, 61)
RED = RGBColor(195, 74, 74)
INK = RGBColor(38, 45, 52)
MUTED = RGBColor(92, 104, 116)
PALE_TEAL = RGBColor(234, 244, 245)
PALE_BLUE = RGBColor(238, 243, 249)
PALE_ORANGE = RGBColor(252, 242, 231)
PALE_GREEN = RGBColor(234, 246, 239)
WHITE = RGBColor(255, 255, 255)
LIGHT_LINE = RGBColor(211, 221, 226)


def iter_shapes(shapes):
    for shape in shapes:
        yield shape
        if shape.shape_type == 6:  # group
            yield from iter_shapes(shape.shapes)


def find_text_shape(slide, exact):
    for shape in iter_shapes(slide.shapes):
        if hasattr(shape, "text") and " ".join(shape.text.split()) == exact:
            return shape
    raise KeyError(f"Text not found on slide: {exact}")


def replace_text(shape, text, size=None, bold=None, font="Times New Roman"):
    tf = shape.text_frame
    tf.clear()
    for idx, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = PP_ALIGN.CENTER if shape.left > Inches(2) else PP_ALIGN.LEFT
        for run in p.runs:
            run.font.name = font
            if size:
                run.font.size = Pt(size)
            if bold is not None:
                run.font.bold = bold


def delete_shape(shape):
    shape._element.getparent().remove(shape._element)


def delete_content(slide, predicate):
    for shape in list(slide.shapes):
        if predicate(shape):
            delete_shape(shape)


def add_box(slide, x, y, w, h, fill=WHITE, line=LIGHT_LINE, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    shp.line.width = Pt(1.2)
    return shp


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    size=22,
    color=INK,
    bold=False,
    align=PP_ALIGN.LEFT,
    font="Calibri",
    valign=MSO_ANCHOR.TOP,
    margin=0.08,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    for idx, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(5)
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
    return box


def add_metric_card(slide, x, y, w, h, value, label, color=TEAL):
    add_box(slide, x, y, w, h, fill=WHITE, line=LIGHT_LINE)
    add_text(
        slide,
        value,
        x + 0.15,
        y + 0.16,
        w - 0.3,
        0.58,
        size=28,
        color=color,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        label,
        x + 0.15,
        y + 0.78,
        w - 0.3,
        h - 0.9,
        size=15,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )


def add_picture_contain(slide, path, x, y, w, h):
    from PIL import Image

    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(
        str(path),
        Inches(x + (w - pw) / 2),
        Inches(y + (h - ph) / 2),
        width=Inches(pw),
        height=Inches(ph),
    )


def add_arrow(slide, x, y, w=0.55, h=0.3):
    arrow = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = TEAL
    arrow.line.color.rgb = TEAL
    return arrow


def set_page_number(slide, value):
    candidates = [
        shape
        for shape in slide.shapes
        if hasattr(shape, "text") and shape.top > Inches(9.8) and shape.text.strip()
    ]
    if candidates:
        page = candidates[-1]
        page.left = Inches(19.18)
        page.width = Inches(0.68)
        replace_text(page, str(value), size=27, font="Calibri")
        for paragraph in page.text_frame.paragraphs:
            paragraph.alignment = PP_ALIGN.RIGHT


def style_plain_textbox(shape, text, size=26, color=INK):
    shape.text_frame.clear()
    shape.text_frame.word_wrap = True
    shape.text_frame.margin_left = Inches(0.05)
    for idx, line in enumerate(text.split("\n")):
        p = shape.text_frame.paragraphs[0] if idx == 0 else shape.text_frame.add_paragraph()
        p_pr = p._p.get_or_add_pPr()
        for bullet_tag in ("a:buChar", "a:buAutoNum", "a:buBlip", "a:buNone"):
            for node in list(p_pr.findall(bullet_tag, p_pr.nsmap)):
                p_pr.remove(node)
        p_pr.insert(0, OxmlElement("a:buNone"))
        p.text = line
        p.space_after = Pt(8)
        p.font.name = "Calibri"
        p.font.size = Pt(size)
        p.font.color.rgb = color


def main():
    prs = Presentation(SOURCE)

    # Cover: align the title with the implemented project (no agent layer in repo).
    cover_title = find_text_shape(
        prs.slides[0],
        "Agent-Based Temporal & Longitudinal Heart Disease Modeling with Deep Learning",
    )
    replace_text(
        cover_title,
        "LONGITUDINAL TEMPORAL DISEASE PROGRESSION\n"
        "Heart Disease Modeling with Deep Learning & Gradient Boosting",
        size=39,
        bold=True,
    )
    for paragraph in cover_title.text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
    cover_title.top = Inches(2.73)
    cover_title.height = Inches(1.85)

    # Contents.
    contents = prs.slides[1]
    replace_text(find_text_shape(contents, "Expected Outcome"), "Project Outcomes & Results", size=26)
    replace_text(find_text_shape(contents, "References"), "References", size=26)
    replace_text(find_text_shape(contents, "5-6"), "5", size=26)
    outcomes_page = find_text_shape(contents, "7")
    replace_text(outcomes_page, "6–10", size=21)
    outcomes_page.left = Inches(8.7)
    outcomes_page.width = Inches(2.1)
    for paragraph in outcomes_page.text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.RIGHT
    references_page = find_text_shape(contents, "8")
    replace_text(references_page, "11", size=21)
    references_page.left = Inches(8.7)
    references_page.width = Inches(2.1)
    for paragraph in references_page.text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.RIGHT

    # Introduction and problem statement.
    intro = find_text_shape(
        prs.slides[2],
        "Most medical AI systems rely on static diagnosis from single-visit data. "
        "Real-world heart diseases evolve over time, not as isolated snapshots. "
        "Clinicians rely on longitudinal trends for accurate decision-making. "
        "This work models heart disease using temporal and longitudinal learning. "
        "Deep sequential models (LSTM, GRU, Transformer) capture patient trajectories. "
        "An agent layer detects risk escalation and abnormal progression patterns. "
        "Goal: Enable intelligent longitudinal modeling beyond static prediction.",
    )
    style_plain_textbox(
        intro,
        "• Cardiovascular deterioration is a trajectory, not a single snapshot.\n"
        "• We formulate progression as binary classification: worsening vs stable/improving.\n"
        "• MIMIC-IV 2.1 ICU admissions are represented as sequences of 6 visits × 59 features.\n"
        "• BiLSTM-attention, BiGRU, and Transformer models learn directly from visit sequences.\n"
        "• Temporal feature engineering exposes trend, variability, velocity, and acceleration.\n"
        "• Calibrated gradient boosting produces the strongest and most deployable model.",
        size=27,
    )
    # Replace legacy agent illustrations with a concise view of the implemented tracks.
    delete_content(prs.slides[2], lambda s: s.name in {"Group 4", "Freeform 13"})
    intro_cards = [
        ("SEQUENCE MODELS", "BiLSTM-attention\nBiGRU\nTransformer encoder", PALE_BLUE, NAVY),
        ("TEMPORAL DESCRIPTORS", "Trend • variability\nvelocity • acceleration\n1,534 rich features", PALE_TEAL, TEAL),
        ("CALIBRATED RISK", "HistGB • LightGBM • XGBoost\nsoft-vote ensemble\ninterpretable importance", PALE_GREEN, GREEN),
    ]
    for i, (title, body, fill, accent) in enumerate(intro_cards):
        x = 2.0 + i * 5.45
        add_box(prs.slides[2], x, 6.45, 4.95, 3.25, fill=fill, line=accent)
        add_text(prs.slides[2], title, x + 0.25, 6.8, 4.45, 0.55, size=21,
                 color=accent, bold=True, align=PP_ALIGN.CENTER)
        add_text(prs.slides[2], body, x + 0.35, 7.55, 4.25, 1.7, size=19,
                 color=INK, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)

    problem = find_text_shape(
        prs.slides[3],
        "Most heart disease prediction systems use static models on single-visit data. "
        "These approaches ignore how cardiovascular conditions evolve over time. "
        "Heart diseases develop progressively & require longitudinal analysis of patient history. "
        "Existing deep learning models capture time patterns but lack interpretability. "
        "They fail to provide actionable insights for clinical decision-making. "
        "Current systems also lack intelligent reasoning to detect risk escalation. "
        "There is a need for an agent-based temporal framework for explainable heart disease prediction.",
    )
    problem_title = find_text_shape(prs.slides[3], "PROBLEM STATEMENT")
    replace_text(problem_title, "PROBLEM STATEMENT", size=40, bold=True, font="Calibri")
    for paragraph in problem_title.text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
    style_plain_textbox(
        problem,
        "• Single-visit classifiers discard clinically meaningful change across admissions.\n"
        "• The target class is imbalanced: only 16.9% of samples represent worsening progression.\n"
        "• Sequence models can rank risk well, but the baseline shows low precision at threshold 0.5.\n"
        "• A useful system must preserve recall while reducing false-positive alerts.\n"
        "• It must also be CPU-friendly, calibrated, and interpretable enough for deployment.\n"
        "Research question: can explicit temporal descriptors outperform heavier sequence models?",
        size=27,
    )

    # Correct the unrelated computer-graphics entry in the inherited survey.
    literature_table = prs.slides[5].shapes[3].table
    row = [
        "2023",
        "MIMIC-IV, a Freely Accessible Electronic Health Record Dataset",
        "A. E. W. Johnson et al.",
        "Presents the deidentified hospital and ICU dataset used for reproducible clinical ML.",
        "Large-scale longitudinal data with detailed documentation.",
        "Credentialed access; observational, single-center data.",
    ]
    for cell, value in zip(literature_table.rows[1].cells, row):
        cell.text = value
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.name = "Times New Roman"
            paragraph.font.size = Pt(17)
            paragraph.alignment = PP_ALIGN.LEFT

    # Methodology slide.
    slide = prs.slides[6]
    delete_content(slide, lambda s: s.name == "Freeform 11")
    set_page_number(slide, 5)
    add_box(slide, 0.95, 2.15, 18.1, 8.2, fill=WHITE, line=LIGHT_LINE)
    steps = [
        ("1", "MIMIC-IV 2.1", "Cardiac cohort\n35,256 samples", PALE_BLUE, NAVY),
        ("2", "Preprocess", "Impute, scale, sanitize\n70/15/15 split", PALE_TEAL, TEAL),
        ("3", "Sequence", "6 visits × 59 features\nper patient sample", PALE_ORANGE, ORANGE),
        ("4", "Two model tracks", "BiLSTM • BiGRU • Transformer\nTemporal GBM features", PALE_GREEN, GREEN),
        ("5", "Validate & calibrate", "Isotonic calibration\nF1-tuned threshold", PALE_BLUE, NAVY),
    ]
    x_positions = [1.25, 4.78, 8.31, 11.84, 15.37]
    for idx, ((num, title, body, fill, accent), x) in enumerate(zip(steps, x_positions)):
        add_box(slide, x, 3.0, 3.0, 3.0, fill=fill, line=accent)
        badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.12), Inches(3.12), Inches(0.48), Inches(0.48))
        badge.fill.solid(); badge.fill.fore_color.rgb = accent; badge.line.color.rgb = accent
        # Re-add number above badge so it is visible.
        add_text(slide, num, x + 0.12, 3.12, 0.48, 0.48, size=17, color=WHITE,
                 bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin=0)
        add_text(slide, title, x + 0.65, 3.12, 2.12, 0.55, size=20, color=accent, bold=True)
        add_text(slide, body, x + 0.25, 4.05, 2.5, 1.45, size=17, color=INK,
                 align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        if idx < 4:
            add_arrow(slide, x + 3.04, 4.3)
    add_box(slide, 2.2, 6.75, 15.6, 2.1, fill=NAVY, line=NAVY)
    add_text(
        slide,
        "Output: calibrated probability of worsening progression\n"
        "Evaluation: Accuracy • Precision • Recall • F1 • ROC-AUC • PR-AUC",
        2.55,
        7.13,
        14.9,
        1.3,
        size=23,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )

    # Outcomes slide.
    slide = prs.slides[7]
    replace_text(find_text_shape(slide, "EXPECTED OUTCOME OF THE PROJECT"), "PROJECT OUTCOMES", size=42, bold=True)
    delete_content(slide, lambda s: s.name in {"Freeform 9", "Freeform 10", "Freeform 12", "TextBox 11"})
    set_page_number(slide, 6)
    add_text(
        slide,
        "The project now delivers a reproducible full-data pipeline and a stronger deployable model.",
        1.3,
        2.38,
        17.4,
        0.7,
        size=26,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    cards = [
        ("35,256", "total longitudinal samples", NAVY),
        ("0.7334", "best F1 • GBM soft vote", TEAL),
        ("0.9462", "best ROC-AUC", GREEN),
        ("3.3 MB", "self-contained deployable artifact", ORANGE),
    ]
    for i, card in enumerate(cards):
        add_metric_card(slide, 1.2 + i * 4.55, 3.45, 4.05, 2.05, *card)
    add_box(slide, 1.2, 6.2, 17.7, 3.15, fill=PALE_BLUE, line=LIGHT_LINE)
    add_text(
        slide,
        "✓ Full MIMIC-IV 2.1 cardiac progression cohort\n"
        "✓ Fair baseline reproduction on identical arrays\n"
        "✓ Calibrated threshold selected only on validation data",
        1.6,
        6.62,
        8.0,
        2.2,
        size=21,
        color=INK,
    )
    add_text(
        slide,
        "✓ Millisecond CPU inference; GPU not required\n"
        "✓ Native feature importance for temporal drivers\n"
        "✓ Saved metrics, predictions, and regeneration scripts",
        10.1,
        6.62,
        8.1,
        2.2,
        size=21,
        color=INK,
    )

    # Dataset/results overview.
    slide = prs.slides[8]
    delete_content(slide, lambda s: s.name == "TextBox 10")
    set_page_number(slide, 7)
    add_box(slide, 0.85, 2.35, 6.0, 7.8, fill=PALE_BLUE, line=LIGHT_LINE)
    add_text(slide, "DATASET & INPUT", 1.2, 2.7, 5.3, 0.55, size=25, color=NAVY, bold=True)
    add_text(
        slide,
        "MIMIC-IV 2.1 cardiac cohort\n"
        "• Train: 24,678 (4,178 worsening)\n"
        "• Validation: 5,289 (896 worsening)\n"
        "• Test: 5,289 (896 worsening)\n"
        "• Positive class: 16.9%\n\n"
        "Each sample: 6 visits × 59 features\n"
        "Vitals • labs • cardiac markers • demographics • history • temporal gaps",
        1.2,
        3.45,
        5.3,
        5.95,
        size=20,
        color=INK,
    )
    add_box(slide, 7.15, 2.35, 11.95, 7.8, fill=WHITE, line=LIGHT_LINE)
    add_text(slide, "BASELINE → IMPROVED", 7.55, 2.64, 11.15, 0.5, size=24, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_picture_contain(slide, ASSETS / "improved_model_comparison.png", 7.45, 3.15, 11.35, 6.65)

    # Feature engineering and deployable pipeline.
    slide = prs.slides[9]
    delete_content(slide, lambda s: s.name == "TextBox 10")
    set_page_number(slide, 8)
    add_box(slide, 0.9, 2.35, 6.3, 7.8, fill=PALE_TEAL, line=LIGHT_LINE)
    add_text(slide, "WHY TEMPORAL GBM WINS", 1.25, 2.7, 5.6, 0.55, size=24, color=TEAL, bold=True)
    add_text(
        slide,
        "The sequence is short (6 timesteps), so explicit change descriptors are highly informative.\n\n"
        "Engineered per feature:\n"
        "• first / last / mean / median\n"
        "• min / max / range / IQR\n"
        "• delta / slope / half-split trend\n"
        "• velocity / acceleration\n"
        "• variability / total variation\n\n"
        "HistGB + LightGBM + XGBoost\n"
        "→ isotonic calibration → soft vote",
        1.25,
        3.45,
        5.55,
        6.25,
        size=19,
        color=INK,
    )
    add_box(slide, 7.5, 2.35, 11.6, 7.8, fill=WHITE, line=LIGHT_LINE)
    add_text(slide, "MOST IMPORTANT TEMPORAL SIGNALS", 7.85, 2.64, 10.9, 0.5, size=23, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_picture_contain(slide, ASSETS / "improved_feature_importance.png", 7.8, 3.18, 10.95, 6.62)

    # Best model and confusion matrix.
    slide = prs.slides[10]
    delete_content(slide, lambda s: s.name in {"Object 9", "TextBox 11"})
    set_page_number(slide, 9)
    add_box(slide, 0.9, 2.35, 6.45, 7.8, fill=NAVY, line=NAVY)
    add_text(slide, "BEST MODEL", 1.3, 2.75, 5.65, 0.55, size=25, color=WHITE, bold=True)
    add_text(slide, "GBM soft-vote ensemble", 1.3, 3.48, 5.65, 0.85, size=29, color=WHITE, bold=True)
    add_text(
        slide,
        "HistGB + LightGBM + XGBoost\n"
        "Isotonic-calibrated\n"
        "Validation-tuned threshold: 0.335",
        1.3,
        4.45,
        5.65,
        1.75,
        size=20,
        color=WHITE,
    )
    add_metric_card(slide, 1.25, 6.55, 2.65, 1.45, "0.7334", "F1", TEAL)
    add_metric_card(slide, 4.25, 6.55, 2.65, 1.45, "0.9462", "ROC-AUC", GREEN)
    add_metric_card(slide, 1.25, 8.35, 2.65, 1.45, "0.7226", "Precision", ORANGE)
    add_metric_card(slide, 4.25, 8.35, 2.65, 1.45, "0.7444", "Recall", RED)
    add_box(slide, 7.65, 2.35, 11.45, 7.8, fill=WHITE, line=LIGHT_LINE)
    add_text(slide, "HELD-OUT TEST SET • n = 5,289", 8.0, 2.68, 10.75, 0.5, size=22, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_picture_contain(slide, ASSETS / "improved_confusion_matrix.png", 8.0, 3.2, 10.7, 6.45)

    # Comparison / interpretation.
    slide = prs.slides[11]
    delete_content(slide, lambda s: s.name in {"Object 9", "TextBox 11"})
    set_page_number(slide, 10)
    add_box(slide, 0.85, 2.35, 5.15, 7.8, fill=PALE_GREEN, line=LIGHT_LINE)
    add_text(slide, "KEY FINDINGS", 1.2, 2.72, 4.45, 0.55, size=25, color=GREEN, bold=True)
    add_text(
        slide,
        "• Best reproduced sequence baseline: Transformer\n"
        "  F1 0.6319 • AUC 0.9168\n\n"
        "• GBM soft vote improves F1 by 0.1015 (+16%).\n\n"
        "• Precision rises from 0.5174 to 0.7226 while recall remains 0.7444.\n\n"
        "• Improved models dominate the ROC and precision-recall curves.\n\n"
        "• Important signals are clinically plausible: SpO₂, blood-pressure and heart-rate variability, bicarbonate, and WBC.",
        1.2,
        3.45,
        4.45,
        6.15,
        size=18,
        color=INK,
    )
    add_box(slide, 6.3, 2.35, 12.8, 7.8, fill=WHITE, line=LIGHT_LINE)
    add_text(slide, "ROC CURVES • BASELINES VS TEMPORAL GBM", 6.65, 2.68, 12.1, 0.5, size=22, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_picture_contain(slide, ASSETS / "improved_roc_curves.png", 6.62, 3.22, 12.15, 6.35)

    # References: replace outdated/general web references with the data and methods used.
    slide = prs.slides[12]
    old_ref = find_text_shape(slide, "[1] “Predicting Disease Progression Using Longitudinal EHR Data with Deep Learning,” ResearchGate, 2025. [Online]. Available: https://www.researchgate.net/publication/392255839 [2] “Predictive Modeling Using Longitudinal Health Data,” Nature Digital Medicine, 2025. [Online]. Available: https://www.nature.com/articles/s41746-025-02296-5 [3] “Deep Learning Approaches for Disease Prediction,” Scientific Reports, vol., 2018. [Online]. Available: https://www.nature.com/articles/s41598-018-36745-x [4] “Machine Learning Models in Healthcare Analytics,” JAMIA Open, vol. 8, no. 2, 2025. [Online]. Available: https://academic.oup.com/jamiaopen/article/8/2/ooaf026/8110091 [5] Y. Si et al., “Deep Representation Learning of Patient Data from Electronic Health Records: A Systematic Review,” IEEE/ACM Trans. Comput. Biol. Bioinf., 2020. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC7615082/")
    style_plain_textbox(
        old_ref,
        "[1] A. E. W. Johnson et al., “MIMIC-IV, a freely accessible electronic health record dataset,” Scientific Data, vol. 10, 2023. DOI: 10.1038/s41597-022-01899-x.\n"
        "[2] A. E. W. Johnson et al., MIMIC-IV v2.1, PhysioNet, 2022. DOI: 10.13026/6mm1-ek67.\n"
        "[3] A. Vaswani et al., “Attention Is All You Need,” Advances in Neural Information Processing Systems, 2017.\n"
        "[4] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” Proc. ACM SIGKDD, 2016.\n"
        "[5] G. Ke et al., “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” NeurIPS, 2017.\n"
        "[6] Project repository: github.com/Venkat-023/Longitudinal-Temporal-Disease-Progression (commit 505e4e5, accessed 12 Aug 2026).",
        size=21,
    )
    set_page_number(slide, 11)

    # Repair inherited overlapping font runs on the closing slide.
    closing = find_text_shape(prs.slides[13], "Thank You!")
    replace_text(closing, "Thank You!", size=54, bold=True, font="Calibri")
    for paragraph in closing.text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER

    # Metadata helps distinguish the revised artifact.
    prs.core_properties.title = "Longitudinal Temporal Disease Progression"
    prs.core_properties.subject = "Updated from repository commit 505e4e5"
    prs.core_properties.comments = (
        "Updated 12 Aug 2026 with full MIMIC-IV results and temporal GBM ensemble."
    )
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
