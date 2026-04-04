#Importing required Libraries
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether
)
from reportlab.platypus import Image as RLImage
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

# ── Colour Palette  ──────────────────────
GREEN_DARK = colors.HexColor('#1b5e20')
GREEN_MID = colors.HexColor('#2e7d32')
GREEN_LIGHT = colors.HexColor('#388e3c')
GREEN_PALE = colors.HexColor('#e8f5e9')
GREEN_ACCENT = colors.HexColor('#a5d6a7')
AMBER_BG = colors.HexColor('#fff8e1')
AMBER_BORDER = colors.HexColor('#f9a825')
AMBER_TEXT = colors.HexColor('#4e342e')
AMBER_STRONG = colors.HexColor('#bf360c')
GREY_LIGHT = colors.HexColor('#f5f5f5')
GREY_BORDER = colors.HexColor('#cfd8dc')
GREY_TEXT = colors.HexColor('#444444')
WHITE = colors.white
BLACK = colors.black

# ── Page dimensions (A4) ────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN_H = 20 * mm
MARGIN_V = 22 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_H


def _build_styles():
    """Return a dict of named ParagraphStyle objects."""
    base = getSampleStyleSheet()

    styles = {
        'title': ParagraphStyle(
            'title',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=WHITE,
            alignment=TA_LEFT,
            leading=28,
        ),
        'subtitle': ParagraphStyle(
            'subtitle',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#c8e6c9'),
            alignment=TA_LEFT,
            leading=14,
        ),
        'section_heading': ParagraphStyle(
            'section_heading',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=GREEN_DARK,
            spaceBefore=4,
            spaceAfter=6,
            leading=16,
        ),
        'field_label': ParagraphStyle(
            'field_label',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8,
            textColor=GREEN_MID,
            leading=11,
            spaceAfter=1,
        ),
        'field_value': ParagraphStyle(
            'field_value',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=BLACK,
            leading=14,
        ),
        'body': ParagraphStyle(
            'body',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            textColor=GREY_TEXT,
            leading=14,
        ),
        'body_bold': ParagraphStyle(
            'body_bold',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9.5,
            textColor=BLACK,
            leading=14,
        ),
        'disclaimer': ParagraphStyle(
            'disclaimer',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            textColor=AMBER_TEXT,
            leading=13,
        ),
        'footer': ParagraphStyle(
            'footer',
            parent=base['Normal'],
            fontName='Helvetica',
            fontSize=8,
            textColor=colors.HexColor('#90a4ae'),
            alignment=TA_CENTER,
            leading=11,
        ),
        'small_caps': ParagraphStyle(
            'small_caps',
            parent=base['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.5,
            textColor=GREEN_MID,
            leading=10,
            spaceAfter=2,
        ),
    }
    return styles


def _header_block(styles: dict) -> list:
    """
    Build the green header banner .
    Returns a list of flowables wrapped in a single-cell Table
    """
    now_str = datetime.now().strftime('%d %B %Y  •  %H:%M')

    title_para = Paragraph('AyurDerma', styles['title'])
    subtitle_para = Paragraph(
        'AI-Powered Ayurvedic Skin Diagnosis Report<br/>'
        f'<font size="8" color="#a5d6a7">Generated: {now_str}</font>',
        styles['subtitle']
    )

    inner = [[title_para], [subtitle_para]]
    t = Table(inner, colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), GREEN_DARK),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 18),
        ('RIGHTPADDING', (0, 0), (-1, -1), 18),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [GREEN_DARK]),
    ]))
    return [t, Spacer(1, 8 * mm)]


def _severity_badge_color(severity: str) -> colors.Color:
    s = severity.lower()
    if s == 'mild':     return colors.HexColor('#2e7d32')
    if s == 'moderate': return colors.HexColor('#e65100')
    return colors.HexColor('#b71c1c')


def _diagnosis_summary_block(data: dict, styles: dict) -> list:
    """
    Three-column info strip: disease | severity | confidence
    """
    disease = str(data.get('disease', 'N/A')).capitalize()
    severity = str(data.get('severity', 'N/A')).capitalize()
    confidence = f"{float(data.get('confidence', 0)):.1f}%"
    sev_color = _severity_badge_color(severity)

    def _cell(label: str, value: str, val_color=GREEN_DARK):
        label_p = Paragraph(label.upper(), styles['small_caps'])
        value_p = Paragraph(
            f'<font name="Helvetica-Bold" size="14" color="{val_color.hexval()}">'
            f'{value}</font>',
            styles['field_value']
        )
        return [label_p, value_p]

    cells = [
        _cell('Detected Condition', disease),
        _cell('Severity', severity, sev_color),
        _cell('Confidence', confidence),
    ]

    t = Table([cells], colWidths=[CONTENT_W / 3] * 3)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), GREEN_PALE),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('BOX', (0, 0), (-1, -1), 1, GREEN_ACCENT),
        ('LINEBEFORE', (1, 0), (1, -1), 1, GREEN_ACCENT),
        ('LINEBEFORE', (2, 0), (2, -1), 1, GREEN_ACCENT),
        ('ROUNDEDCORNERS', [4]),
    ]))
    return [t, Spacer(1, 6 * mm)]


def _patient_block(data: dict, styles: dict) -> list:
    """Patient profile section."""
    patient = data.get('patient_data', {})
    age = patient.get('age', 'N/A')
    s_type = str(patient.get('skinType', 'N/A')).capitalize()
    s_sens = str(patient.get('skinSensitivity', 'N/A')).capitalize()

    heading = Paragraph('Patient Profile', styles['section_heading'])
    hr = HRFlowable(width=CONTENT_W, thickness=1,
                    color=GREEN_ACCENT, spaceAfter=5)

    rows = [
        [Paragraph('Age', styles['field_label']),
         Paragraph(str(age), styles['field_value'])],
        [Paragraph('Skin Type', styles['field_label']),
         Paragraph(s_type, styles['field_value'])],
        [Paragraph('Skin Sensitivity', styles['field_label']),
         Paragraph(s_sens, styles['field_value'])],
    ]
    t = Table(rows, colWidths=[50 * mm, CONTENT_W - 50 * mm])
    t.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ]))
    return [heading, hr, t, Spacer(1, 5 * mm)]


def _treatment_row(label: str, value: str, styles: dict,
                   highlight: bool = False) -> list:
    """Single treatment detail row (two-column table)."""
    bg = GREEN_PALE if highlight else GREY_LIGHT
    lp = Paragraph(label.upper(), styles['small_caps'])
    vp = Paragraph(str(value) if value else 'N/A', styles['body'])
    row_table = Table([[lp, vp]],
                      colWidths=[46 * mm, CONTENT_W - 46 * mm])
    row_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (0, -1), 10),
        ('LEFTPADDING', (1, 0), (1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('LINEABOVE', (0, 0), (-1, 0), 0.5, GREY_BORDER),
    ]))
    return [row_table, Spacer(1, 1.5 * mm)]


def _treatment_block(data: dict, styles: dict) -> list:
    """Full recommended treatment section."""
    t_data = data.get('treatment', {})
    heading = Paragraph('Recommended Ayurvedic Treatment', styles['section_heading'])
    hr = HRFlowable(width=CONTENT_W, thickness=1,
                    color=GREEN_ACCENT, spaceAfter=6)

    flowables = [heading, hr]

    # Selected herb (highlighted)
    herb_name = t_data.get('selected_herb') or t_data.get('herb_english', 'N/A')
    flowables += _treatment_row('Selected Herb', herb_name, styles, highlight=True)
    flowables += _treatment_row('English Name', t_data.get('herb_english', 'N/A'), styles)
    flowables += _treatment_row('Sinhala Name', t_data.get('herb_sinhala', 'N/A'), styles)
    flowables += _treatment_row('Scientific Name', t_data.get('herb_scientific', 'N/A'), styles)
    flowables += _treatment_row('Part Used', t_data.get('herb_part_used', 'N/A'), styles)
    flowables += _treatment_row('Preparation', t_data.get('preparation', 'N/A'), styles)
    flowables += _treatment_row('Application', t_data.get('application', 'N/A'), styles)
    flowables += _treatment_row('Frequency', t_data.get('frequency', 'N/A'), styles)
    flowables += _treatment_row('Duration', t_data.get('duration', 'N/A'), styles)
    flowables += _treatment_row('Precautions', t_data.get('precautions', 'N/A'), styles)
    flowables += _treatment_row('When to See Doctor', t_data.get('when_to_see_doctor', 'N/A'), styles)

    flowables.append(Spacer(1, 5 * mm))
    return flowables


def _disclaimer_block(styles: dict) -> list:
    """Amber disclaimer box — matches about.html disclaimer style."""
    text = (
        '<b>Medical Disclaimer:</b> This report is generated by AyurDerma, a research '
        'prototype for educational and preliminary guidance purposes only. It is not a '
        'certified medical device and should not replace professional medical advice, '
        'diagnosis or treatment. Always consult a qualified health professional for any '
        'medical condition. Perform a patch test before applying any herbal remedy.'
    )
    p = Paragraph(text, styles['disclaimer'])
    t = Table([[p]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), AMBER_BG),
        ('BOX', (0, 0), (-1, -1), 1.5, AMBER_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    return [t, Spacer(1, 4 * mm)]


def _footer_block(styles: dict) -> list:
    text = '© 2025 AyurDerma  •  Traditional Healing, Modern Technology  •  Research Prototype'
    return [
        HRFlowable(width=CONTENT_W, thickness=0.5, color=GREY_BORDER, spaceBefore=4),
        Paragraph(text, styles['footer']),
    ]
