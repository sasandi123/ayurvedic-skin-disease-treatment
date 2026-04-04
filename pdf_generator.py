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

