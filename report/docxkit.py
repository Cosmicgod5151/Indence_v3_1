"""Small python-docx toolkit that enforces the CSET400C §12 report format.

Times New Roman throughout; body 12 pt, 1.5 spacing, justified, A4.
Chapter heading 16 pt bold uppercase centred; main heading 14 pt bold;
subheading 12 pt bold; sub-subheading 12 pt bold italic; captions 11 pt;
table text 10 pt.
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Cm

FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)


def _fonts(rpr):
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)


def _style(doc, name, size, bold=False, italic=False, align=None, before=0, after=6, spacing=1.5,
           base=None, keep_next=False, outline=None, caps=False):
    styles = doc.styles
    try:
        st = styles[name]
    except KeyError:
        from docx.enum.style import WD_STYLE_TYPE
        st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        if base:
            st.base_style = styles[base]
    f = st.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = BLACK
    f.all_caps = caps
    _fonts(st.element.get_or_add_rPr())
    pf = st.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = spacing
    pf.keep_with_next = keep_next
    if align is not None:
        pf.alignment = align
    if outline is not None:
        ppr = st.element.get_or_add_pPr()
        ol = ppr.find(qn("w:outlineLvl"))
        if ol is None:
            ol = OxmlElement("w:outlineLvl")
            ppr.append(ol)
        ol.set(qn("w:val"), str(outline))
    return st


def _field(paragraph, instr, placeholder=""):
    def el(tag, **attrs):
        e = OxmlElement(tag)
        for k, v in attrs.items():
            e.set(qn(k), v)
        return e
    r1 = paragraph.add_run()
    r1._r.append(el("w:fldChar", **{"w:fldCharType": "begin", "w:dirty": "true"}))
    r2 = paragraph.add_run()
    it = el("w:instrText", **{"xml:space": "preserve"})
    it.text = f" {instr} "
    r2._r.append(it)
    r3 = paragraph.add_run()
    r3._r.append(el("w:fldChar", **{"w:fldCharType": "separate"}))
    paragraph.add_run(placeholder)
    r5 = paragraph.add_run()
    r5._r.append(el("w:fldChar", **{"w:fldCharType": "end"}))


def _shade(cell, hex_fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcpr.append(shd)


class Report:
    def __init__(self):
        self.doc = Document()
        d = self.doc
        sec = d.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = Inches(1)
        sec.top_margin = sec.bottom_margin = Inches(1)
        J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
        _style(d, "Normal", 12, align=J, after=6)
        _style(d, "Heading 1", 16, bold=True, align=C, before=0, after=18, keep_next=True, caps=True)
        _style(d, "Heading 2", 14, bold=True, align=L, before=12, after=6, keep_next=True)
        _style(d, "Heading 3", 12, bold=True, align=L, before=10, after=4, keep_next=True)
        _style(d, "Heading 4", 12, bold=True, italic=True, align=L, before=8, after=4, keep_next=True)
        _style(d, "Front Heading", 16, bold=True, align=C, after=18, base="Normal", keep_next=True, caps=True)
        _style(d, "Figure Caption", 11, align=C, before=4, after=12, spacing=1.0, base="Normal")
        _style(d, "Table Caption", 11, align=C, before=10, after=4, spacing=1.0, base="Normal", keep_next=True)
        _style(d, "Table Text", 10, align=L, after=0, spacing=1.0, base="Normal")
        _style(d, "Field Holder", 1, align=L, after=0, spacing=1.0, base="Normal")
        _style(d, "Equation", 12, italic=True, align=C, before=4, after=8, base="Normal")
        for s in ("List Bullet", "List Number"):
            _style(d, s, 12, align=J, after=3)
            d.styles[s].paragraph_format.left_indent = Inches(0.35)
        for s in ("toc 1", "toc 2", "toc 3", "table of figures"):
            try:
                _style(d, s, 12, after=0, spacing=1.0)
            except Exception:
                pass
        self.chapter_no = 0
        self.fig_no = 0
        self.tab_no = 0
        self.prefix = ""

    # ------------------------------------------------------------ sections
    def new_section(self, fmt, start=1, header_text=None):
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), fmt)
        pg.set(qn("w:start"), str(start))
        sec._sectPr.append(pg)
        sec.footer.is_linked_to_previous = False
        sec.header.is_linked_to_previous = False
        fp = sec.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _field(fp, "PAGE", "1")
        for r in fp.runs:
            r.font.name, r.font.size = FONT, Pt(11)
        hp = sec.header.paragraphs[0]
        hp.text = header_text or ""
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for r in hp.runs:
            r.font.name, r.font.size, r.font.italic = FONT, Pt(9), True
        return sec

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ------------------------------------------------------------ text
    def _runs(self, p, text, size=None):
        # **bold** and _italic_ markup
        for part in re.split(r"(\*\*[^*]+\*\*|__[^_]+__)", text):
            if not part:
                continue
            if part.startswith("**"):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("__"):
                r = p.add_run(part[2:-2]); r.italic = True
            else:
                r = p.add_run(part)
            if size:
                r.font.size = Pt(size)
        return p

    def p(self, text, style="Normal", align=None, size=None, after=None):
        para = self.doc.add_paragraph(style=style)
        self._runs(para, text, size)
        if align is not None:
            para.alignment = align
        if after is not None:
            para.paragraph_format.space_after = Pt(after)
        return para

    def bullets(self, items, style="List Bullet"):
        for it in items:
            self.p(it, style=style)

    def eq(self, text):
        return self.p(text, style="Equation")

    def front_heading(self, text, toc=False, new_page=True):
        if toc:
            para = self.doc.add_paragraph(style="Heading 1")
        else:
            para = self.doc.add_paragraph(style="Front Heading")
        para.add_run(text)
        para.paragraph_format.page_break_before = new_page
        return para

    def chapter(self, title, first=False):
        self.chapter_no += 1
        self.fig_no = self.tab_no = 0
        self.prefix = str(self.chapter_no)
        para = self.doc.add_paragraph(style="Heading 1")
        para.add_run(f"Chapter {self.chapter_no}: {title}")
        para.paragraph_format.page_break_before = not first
        return para

    def appendix(self, letter, title):
        self.fig_no = self.tab_no = 0
        self.prefix = letter
        para = self.doc.add_paragraph(style="Heading 1")
        para.add_run(f"Appendix {letter}: {title}")
        para.paragraph_format.page_break_before = True

    def h2(self, t):
        return self.doc.add_paragraph(t, style="Heading 2")

    def h3(self, t):
        return self.doc.add_paragraph(t, style="Heading 3")

    def h4(self, t):
        return self.doc.add_paragraph(t, style="Heading 4")

    # ------------------------------------------------------------ figures / tables
    def figure(self, path, caption, width=6.0):
        para = self.doc.add_paragraph(style="Normal")
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.line_spacing = 1.0
        para.add_run().add_picture(str(path), width=Inches(width))
        self.fig_no += 1
        cap = self.doc.add_paragraph(style="Figure Caption")
        r = cap.add_run(f"Figure {self.prefix}.{self.fig_no}: ")
        r.bold = True
        self._runs(cap, caption)
        return f"{self.prefix}.{self.fig_no}"

    def next_fig(self):
        return f"{self.prefix}.{self.fig_no + 1}"

    def next_tab(self):
        return f"{self.prefix}.{self.tab_no + 1}"

    def table(self, header, rows, widths, caption, size=10, bold_first_col=False):
        if caption:
            self.tab_no += 1
            cap = self.doc.add_paragraph(style="Table Caption")
            r = cap.add_run(f"Table {self.prefix}.{self.tab_no}: ")
            r.bold = True
            self._runs(cap, caption)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            self._runs(c.paragraphs[0], f"**{h}**", size)
            c.paragraphs[0].style = self.doc.styles["Table Text"]
            _shade(c, "D9E2F3")
        trpr = t.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trpr.append(th)
        for row in rows:
            cells = t.add_row().cells
            for i, val in enumerate(row):
                c = cells[i]
                c.text = ""
                txt = str(val)
                if bold_first_col and i == 0 and not txt.startswith("**"):
                    txt = f"**{txt}**"
                lines = txt.split("\n")
                c.paragraphs[0].style = self.doc.styles["Table Text"]
                self._runs(c.paragraphs[0], lines[0], size)
                for ln in lines[1:]:
                    para = c.add_paragraph(style="Table Text")
                    self._runs(para, ln, size)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
        # cell padding
        tblpr = t._tbl.tblPr
        mar = OxmlElement("w:tblCellMar")
        for side, v in (("top", 25), ("bottom", 25), ("left", 70), ("right", 70)):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); mar.append(e)
        tblpr.append(mar)
        # keep small tables on one page, and never strand a header row
        n = len(t.rows)
        keep_rows = n - 1 if n <= 16 else 2
        for ri, row in enumerate(t.rows):
            trpr2 = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trpr2.append(cs)
            if ri < keep_rows:
                for c in row.cells:
                    for para in c.paragraphs:
                        para.paragraph_format.keep_with_next = True
        # small gap after the table
        gap = self.doc.add_paragraph(style="Normal")
        gap.paragraph_format.space_after = Pt(4)
        gap.paragraph_format.line_spacing = 1.0
        return t

    def toc(self, instr, placeholder="Right-click and choose Update Field to build this list."):
        para = self.doc.add_paragraph(style="Field Holder")
        _field(para, instr, placeholder)

    def save(self, path):
        self.doc.save(str(path))
