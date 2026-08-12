from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor
from docx.text.run import Run


DOCX_PATH = Path(__file__).with_name("paper.zh.docx")
TOTAL_TABLE_WIDTH = 9072
TABLE_WIDTHS = (
    (1600, 1400, 1400, 2200, 2472),
    (1400, 1200, 3500, 2972),
    (1100, 1450, 1450, 1450, 1700, 1922),
    (1000, 3200, 3200, 1672),
    (1800, 2400, 2400, 2472),
)


def style_by_name(doc, name):
    for style in doc.styles:
        if style.name == name:
            return style
    raise KeyError(f"style not found: {name}")


def set_run_fonts(run, cjk="SimSun", latin="Times New Roman", size=None, bold=None):
    run.font.name = latin
    run.font.color.rgb = RGBColor(0, 0, 0)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), latin)
    fonts.set(qn("w:hAnsi"), latin)
    fonts.set(qn("w:eastAsia"), cjk)
    fonts.set(qn("w:cs"), latin)


def set_style_font(style, cjk, latin, size, bold=None):
    style.font.name = latin
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        style.font.bold = bold
    rpr = style._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), latin)
    fonts.set(qn("w:hAnsi"), latin)
    fonts.set(qn("w:eastAsia"), cjk)
    fonts.set(qn("w:cs"), latin)


def set_cell_margins(cell, top=50, start=120, bottom=50, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width))
    tc_w.set(qn("w:type"), "dxa")


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "B7B7B7")


def equal_table_widths(table):
    column_count = len(table.columns)
    if column_count == 0 and table.rows:
        column_count = max(len(row.cells) for row in table.rows)
    if column_count <= 0:
        raise ValueError("cannot style a table without cells")
    base_width, remainder = divmod(TOTAL_TABLE_WIDTH, column_count)
    widths = [base_width] * column_count
    widths[-1] += remainder
    return tuple(widths)


def set_table_geometry(table, widths, *, font_size=8.5, cell_margin=50):
    if not widths:
        raise ValueError("table widths must not be empty")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr

    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(TOTAL_TABLE_WIDTH))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row_index, row in enumerate(table.rows):
        if row_index == 0:
            tr_pr = row._tr.get_or_add_trPr()
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            tr_pr.append(repeat)
        for col_index, cell in enumerate(row.cells):
            width = widths[min(col_index, len(widths) - 1)]
            set_cell_width(cell, width)
            set_cell_margins(cell, top=cell_margin, bottom=cell_margin)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index == 0:
                shade_cell(cell, "F2F4F7")
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.first_line_indent = Pt(0)
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    set_run_fonts(
                        run,
                        size=font_size,
                        bold=True if row_index == 0 else None,
                    )
    set_table_borders(table)


def iter_all_runs(paragraph):
    for run_element in paragraph._p.xpath(".//w:r"):
        yield Run(run_element, paragraph)


def replace_equation_cross_reference(paragraph):
    replacements = {
        "eq:gate": "上述门控关系",
        "eq:quality": "上述质量目标",
    }
    for anchor, replacement in replacements.items():
        for hyperlink in paragraph._p.xpath(
            f'.//w:hyperlink[@w:anchor="{anchor}"]'
        ):
            previous = hyperlink.getprevious()
            while previous is not None:
                text_nodes = previous.xpath(".//w:t")
                if text_nodes:
                    text = text_nodes[-1].text or ""
                    if text.endswith("式"):
                        text_nodes[-1].text = text[:-1]
                    break
                previous = previous.getprevious()

            text_nodes = hyperlink.xpath(".//w:t")
            if text_nodes:
                text_nodes[0].text = replacement
                for node in text_nodes[1:]:
                    node.text = ""


def number_captions(doc):
    counters = {"Table Caption": 0, "Image Caption": 0}
    prefixes = {"Table Caption": "表", "Image Caption": "图"}
    for paragraph in doc.paragraphs:
        style_name = paragraph.style.name
        if style_name not in counters:
            continue
        counters[style_name] += 1
        prefix = f"{prefixes[style_name]}{counters[style_name]} "
        if paragraph.text.startswith(prefix.strip()):
            continue
        if paragraph.runs:
            paragraph.runs[0].text = prefix + paragraph.runs[0].text
        else:
            paragraph.add_run(prefix)


def append_page_field(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_fonts(run, size=9)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    value = OxmlElement("w:t")
    value.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, separate, value, end))


def main():
    doc = Document(DOCX_PATH)

    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.top_margin = Mm(24)
        section.bottom_margin = Mm(24)
        section.left_margin = Mm(25)
        section.right_margin = Mm(25)
        section.header_distance = Mm(12.5)
        section.footer_distance = Mm(12.5)
        footer = section.footer
        footer.is_linked_to_previous = False
        footer_paragraph = footer.paragraphs[0]
        footer_paragraph.clear()
        append_page_field(footer_paragraph)

    body_styles = ("Normal", "Body Text", "First Paragraph", "Abstract", "Compact")
    for name in body_styles:
        style = style_by_name(doc, name)
        set_style_font(style, "SimSun", "Times New Roman", 10.5)
        pf = style.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.first_line_indent = Pt(21)
        pf.space_before = Pt(0)
        pf.space_after = Pt(6)
        pf.line_spacing = 1.25
        pf.widow_control = True

    title = style_by_name(doc, "Title")
    set_style_font(title, "SimHei", "Times New Roman", 18, True)
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.first_line_indent = Pt(0)
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(12)
    title.paragraph_format.keep_with_next = True

    heading_tokens = {
        "Heading 1": ("SimHei", 14, 14, 6),
        "Heading 2": ("SimHei", 12, 10, 4),
        "Heading 3": ("SimHei", 11, 8, 4),
    }
    for name, (cjk, size, before, after) in heading_tokens.items():
        style = style_by_name(doc, name)
        set_style_font(style, cjk, "Times New Roman", size, True)
        pf = style.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.line_spacing = 1.0
        pf.keep_with_next = True
        pf.keep_together = True

    for name in ("Caption", "Table Caption", "Image Caption"):
        style = style_by_name(doc, name)
        set_style_font(style, "SimSun", "Times New Roman", 9)
        style.font.italic = False
        pf = style.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(3)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.0
        pf.keep_with_next = name == "Table Caption"
        pf.keep_together = True

    hyperlink = style_by_name(doc, "Hyperlink")
    set_style_font(hyperlink, "SimSun", "Times New Roman", 10.5)
    hyperlink.font.underline = False

    bibliography = style_by_name(doc, "Bibliography")
    set_style_font(bibliography, "SimSun", "Times New Roman", 9)
    bibliography.paragraph_format.left_indent = Pt(18)
    bibliography.paragraph_format.first_line_indent = Pt(-18)
    bibliography.paragraph_format.space_after = Pt(3)
    bibliography.paragraph_format.line_spacing = 1.0

    number_captions(doc)

    for paragraph in doc.paragraphs:
        replace_equation_cross_reference(paragraph)
        if paragraph.text.startswith("关键词："):
            paragraph.paragraph_format.first_line_indent = Pt(0)
        if paragraph.style.name == "Figure":
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.space_before = Pt(6)
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.keep_with_next = True
        if paragraph._p.xpath(".//m:oMathPara"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(4)
        for run in iter_all_runs(paragraph):
            if paragraph.style.name == "Title":
                set_run_fonts(run, "SimHei", "Times New Roman", 18, True)
            elif paragraph.style.name.startswith("Heading"):
                size = {"Heading 1": 14, "Heading 2": 12, "Heading 3": 11}.get(
                    paragraph.style.name, 10.5
                )
                set_run_fonts(run, "SimHei", "Times New Roman", size, True)
            elif paragraph.style.name in {"Caption", "Table Caption", "Image Caption"}:
                set_run_fonts(run, size=9)
                run.font.italic = False
            elif paragraph.style.name == "Bibliography":
                set_run_fonts(run, size=9)
            else:
                set_run_fonts(run, size=10.5)

    for index, table in enumerate(doc.tables):
        widths = (
            TABLE_WIDTHS[index]
            if index < len(TABLE_WIDTHS)
            else equal_table_widths(table)
        )
        compact_seed_table = index == 2
        set_table_geometry(
            table,
            widths,
            font_size=8.0 if compact_seed_table else 8.5,
            cell_margin=25 if compact_seed_table else 50,
        )

    for shape in doc.inline_shapes:
        max_width = Mm(156)
        if shape.width > max_width:
            ratio = max_width / shape.width
            shape.width = max_width
            shape.height = int(shape.height * ratio)

    doc.core_properties.title = "面向图像条件软传感器退化的可靠多模态垃圾分类方法"
    doc.core_properties.subject = "多模态垃圾分类论文"
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.save(DOCX_PATH)


if __name__ == "__main__":
    main()
