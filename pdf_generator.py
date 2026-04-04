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

