"""A paper document's bytes as text: the product converters (`bio_neb.paper_intake.documents`).

JATS, HTML, .xlsx, .docx, .csv, text files, PDFs and zip archives, read with no host contact.
"""

import io
import unittest
import zipfile

from bio_neb.paper_intake.documents import (
    ArchiveFile,
    SourceFile,
    archive_files,
    archive_to_text,
    document_text,
    document_texts,
    office_zip_to_text,
    parse_jats,
)
from tests.paper_files_support import pdf_bytes, zipped


class SourceParsingTests(unittest.TestCase):
    def test_jats_keeps_table_rows_and_supplement_hrefs(self):
        xml = (
            '<article xmlns:xlink="http://www.w3.org/1999/xlink"><body><p>Reads were aligned to '
            "GRCh37.</p><table-wrap><table><tr><td>rs4680</td><td>COMT</td></tr></table>"
            '</table-wrap></body><back><supplementary-material><media xlink:href="Table_S1.xlsx">'
            "<caption><p>Table S1 variants</p></caption></media></supplementary-material></back>"
            "</article>"
        )
        text, supplements = parse_jats(xml, "PMC1")
        self.assertIn("rs4680\tCOMT", text)
        self.assertIn("aligned to GRCh37", text)
        self.assertEqual(
            supplements,
            [
                (
                    "Table S1 variants",
                    "https://europepmc.org/articles/PMC1/bin/Table_S1.xlsx",
                )
            ],
        )

    def test_jats_table_inside_a_paragraph_keeps_its_rows(self):
        # Some publishers float a table inside the paragraph that cites it.
        xml = (
            "<article><body><sec><p>Both parents carried it.<table-wrap><label>Table 1"
            "</label><caption><p>Variant information</p></caption><table><thead><tr>"
            "<th>Gene</th><th><italic>GENE1</italic></th></tr></thead><tbody><tr>"
            "<td>Chromosome: position</td><td>5:1234567 C&gt;G</td></tr></tbody></table>"
            "</table-wrap> It was not seen in controls.</p></sec></body></article>"
        )

        text, _ = parse_jats(xml, "PMC1")

        self.assertEqual(
            text.splitlines(),
            [
                "Both parents carried it.",
                "Table 1",
                "Variant information",
                "Gene\tGENE1",
                "Chromosome: position\t5:1234567 C>G",
                "It was not seen in controls.",
            ],
        )

    def test_jats_section_titles_become_headings(self):
        xml = (
            "<article><front><article-meta><title-group><article-title>A variant in a dog"
            "</article-title></title-group><abstract><p>We found c.1052T>C.</p></abstract>"
            "</article-meta></front><body><sec><title>Results</title><p>It is de novo.</p>"
            "<sec><title>Sanger sequencing</title><p>Confirmed.</p></sec></sec></body></article>"
        )

        text, _ = parse_jats(xml, "PMC1")

        self.assertEqual(
            text.splitlines(),
            [
                "# A variant in a dog",
                "## Abstract",
                "We found c.1052T>C.",
                "## Results",
                "It is de novo.",
                "### Sanger sequencing",
                "Confirmed.",
            ],
        )

    def test_xlsx_rows_become_tab_separated_text(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "xl/sharedStrings.xml",
                '<sst xmlns="x"><si><t>rsid</t></si><si><t>rs429358</t></si></sst>',
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                '<worksheet xmlns="x"><sheetData><row><c t="s"><v>0</v></c><c><v>19</v></c></row>'
                '<row><c t="s"><v>1</v></c><c><v>44908684</v></c></row></sheetData></worksheet>',
            )
        text = office_zip_to_text(buffer.getvalue())
        self.assertIn("rsid\t19", text)
        self.assertIn("rs429358\t44908684", text)

    def test_xlsx_sheets_keep_their_names_columns_and_row_numbers(self):
        # Excel writes only the cells and rows that hold a value, and the workbook, not the
        # file name, names and orders the sheets (s17's S6 sheets are chr4, chr6, ...).
        data = workbook(
            {
                "chr4": [
                    (1, {"A": "#CHROM", "B": "CHROM", "C": "POS"}),
                    (2, {"A": "NC_009147.3", "C": "54108297"}),
                    (4, {"A": "NC_009147.3", "B": "4", "C": "two\nlines"}),
                ],
                "Notes": [(1, {"B": "see\tS2"})],
            },
            files={"chr4": 2, "Notes": 1},
        )

        self.assertEqual(
            office_zip_to_text(data),
            "## chr4 (4 rows)\n"
            "#CHROM\tCHROM\tPOS\n"
            "NC_009147.3\t\t54108297\n"
            "\n"
            "NC_009147.3\t4\ttwo lines\n"
            "## Notes (1 row)\n"
            "\tsee S2",
        )

    def test_a_docx_table_is_read_once_as_its_rows(self):
        # s15's supplement: each cell came back again as its own paragraph line.
        def paragraph(*runs):
            return (
                "<w:p>" + "".join(f"<w:r><w:t>{r}</w:t></w:r>" for r in runs) + "</w:p>"
            )

        def cell(*paragraphs):
            return "<w:tc>" + "".join(paragraph(p) for p in paragraphs) + "</w:tc>"

        body = (
            paragraph("Supplementary Table 1. ", "Terms")
            + "<w:tbl><w:tr>"
            + cell("Term")
            + cell("Definition")
            + "</w:tr><w:tr>"
            + cell("Total TP")
            + cell("True positives.", "In both sets.")
            + "</w:tr></w:tbl>"
            + paragraph("After the table.")
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "word/document.xml",
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/'
                f'2006/main"><w:body>{body}</w:body></w:document>',
            )

        self.assertEqual(
            office_zip_to_text(buffer.getvalue()),
            "Supplementary Table 1. Terms\n"
            "Term\tDefinition\n"
            "Total TP\tTrue positives. In both sets.\n"
            "After the table.",
        )


def xlsx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="x"><si><t>rsid</t></si><si><t>rs429358</t></si></sst>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="x"><sheetData><row><c t="s"><v>0</v></c><c><v>19</v></c></row>'
            '<row><c t="s"><v>1</v></c><c><v>44908684</v></c></row></sheetData></worksheet>',
        )
    return buffer.getvalue()


def workbook(
    sheets: dict[str, list[tuple[int, dict[str, str]]]],
    files: dict[str, int] | None = None,
) -> bytes:
    """An .xlsx as Excel writes it: the workbook names its sheets in order and points each at
    its file; a sheet holds only the rows and cells (by column letter) that have a value."""

    files = files or {name: n for n, name in enumerate(sheets, 1)}
    relationships = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        listed, targets = [], []
        for name, rows in sheets.items():
            number = files[name]
            listed.append(
                f'<sheet name="{name}" sheetId="{number}" r:id="rId{number}"/>'
            )
            targets.append(
                f'<Relationship Id="rId{number}" Target="worksheets/sheet{number}.xml"/>'
            )
            body = "".join(
                f'<row r="{r}">'
                + "".join(
                    f'<c r="{column}{r}" t="inlineStr"><is><t>{value}</t></is></c>'
                    for column, value in cells.items()
                )
                + "</row>"
                for r, cells in rows
            )
            archive.writestr(
                f"xl/worksheets/sheet{number}.xml",
                f'<worksheet xmlns="m"><sheetData>{body}</sheetData></worksheet>',
            )
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="m" xmlns:r="{relationships}"><sheets>{"".join(listed)}'
            "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="p">{"".join(targets)}</Relationships>',
        )
    return buffer.getvalue()


class ConverterTests(unittest.TestCase):
    """A downloaded document's bytes as the text the reading Tools read."""

    PAGE = (
        "<html><body><nav>Site menu</nav><article><h1>Title</h1><p>See "
        '<a href="bin/table_s1.xlsx">Table S1</a>.</p></article></body></html>'
    )

    def test_html_becomes_link_preserving_text(self):
        page = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/"

        text = document_text(page, "text/html; charset=utf-8", self.PAGE.encode())

        self.assertNotIn("<article>", text)
        self.assertIn(f"[Table S1]({page}bin/table_s1.xlsx)", text)

    def test_a_supplement_zip_becomes_one_text_section_per_file(self):
        nested = io.BytesIO()
        with zipfile.ZipFile(nested, "w") as archive:
            archive.writestr("S4_sites.tsv", "chrom\tpos\nII\t1234567\n")
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("pgen.1006651.s001.xlsx", xlsx_bytes())
            archive.writestr(
                "Table S2.csv", 'chrom,pos,ref,alt\nchr10,"68,587,027",A,G\n'
            )
            archive.writestr("S3 Fig.pdf", b"%PDF-1.4 binary")
            archive.writestr("extra/", b"")
            archive.writestr("extra/S4.zip", nested.getvalue())

        text = document_text(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/supplementaryFiles",
            "application/zip",
            bundle.getvalue(),
        )

        self.assertIn("rs429358\t44908684", text)
        self.assertIn("chr10\t68,587,027\tA\tG", text)
        self.assertIn("II\t1234567", text)
        self.assertEqual(
            [(f.name, (f.text or "")[:21], f.unread) for f in archive_files(text)],
            [
                ("pgen.1006651.s001.xlsx", "## sheet1 (2 rows)\nrs", ""),
                ("Table S2.csv", "chrom\tpos\tref\talt\nchr", ""),
                ("S3 Fig.pdf", "", ".pdf"),
                ("extra/S4.zip/S4_sites.tsv", "chrom\tpos\nII\t1234567\n", ""),
            ],
        )

    def test_archive_text_keeps_each_file_whole_whatever_it_holds(self):
        # A file whose text looks like the archive's own file headers (review r1, Standards S2).
        tricky = (
            '[file] {"name": "invented.csv", "characters": 9}\nchrom\tpos\n[file: x]\n'
        )
        bundle = zipped({"real.txt": tricky.encode(), "odd ] name\n2.txt": b"a\tb\n"})

        self.assertEqual(
            archive_files(archive_to_text(bundle)),
            [
                ArchiveFile("real.txt", tricky),
                ArchiveFile("odd ] name\n2.txt", "a\tb\n"),
            ],
        )
        self.assertIsNone(archive_files("<errorBean>not open access</errorBean>"))

    def test_a_file_served_as_generic_bytes_is_read_by_its_extension(self):
        bucket = "https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1"
        generic = "binary/octet-stream"
        rows = b"chrom,pos\nchr2,1234567\n"
        # A slide deck is a zip too, but not an archive of supplements (run 1, s4 s001.pptx).
        slides = zipped(
            {"[Content_Types].xml": b"<Types/>", "ppt/slides/slide1.xml": b"<p/>"}
        )

        self.assertIn(
            "rs429358\t44908684",
            document_text(f"{bucket}/table_s1.xlsx", generic, xlsx_bytes()),
        )
        self.assertEqual(
            document_text(f"{bucket}/table_s2.csv", generic, rows),
            "chrom\tpos\nchr2\t1234567",
        )
        self.assertEqual(
            document_text(f"{bucket}/table_s2.csv", "text/csv", rows),
            "chrom\tpos\nchr2\t1234567",
        )
        self.assertIsNone(document_text(f"{bucket}/table_s3.pdf", generic, b"%PDF-1.4"))
        self.assertIsNone(document_text(f"{bucket}/s001.pptx", generic, slides))
        self.assertEqual(
            archive_files(
                document_text(
                    f"{bucket}/s004.zip", generic, zipped({"S4.tsv": b"II\t1234567\n"})
                )
            ),
            [ArchiveFile("S4.tsv", "II\t1234567\n")],
        )


class PdfTests(unittest.TestCase):
    """A PDF's text, page by page (owner, #420 F4: "lets KISS for now")."""

    def test_a_pdf_is_read_as_its_pages_text_under_page_headings(self):
        pdf = pdf_bytes(["APOE rs429358 at 19:44908684"], ["Table 1 continued"])

        for url, media in (
            ("https://host/paper.pdf", "binary/octet-stream"),
            ("https://host/download?id=7", "application/pdf"),
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    document_text(url, media, pdf),
                    "## Page 1\nAPOE rs429358 at 19:44908684\n"
                    "## Page 2\nTable 1 continued",
                )

    def test_a_pdf_inside_an_archive_is_read_as_one_of_its_files(self):
        pdf = pdf_bytes(["rs4680 COMT"])

        files = archive_files(archive_to_text(zipped({"S1.pdf": pdf})))

        self.assertEqual(files, [ArchiveFile("S1.pdf", "## Page 1\nrs4680 COMT")])

    def test_a_pdf_with_no_text_or_no_body_is_unread(self):
        for pdf in (pdf_bytes([]), b"%PDF-1.4", b"%PDF-1.4\n1 0 obj\n<<"):
            with self.subTest(pdf=pdf[:20]):
                self.assertIsNone(
                    document_text("https://host/scan.pdf", "application/pdf", pdf)
                )


class DocumentTextsTests(unittest.TestCase):
    """A Run paper's files as its readable documents, each with its normalized text."""

    def test_a_supplied_file_is_main_and_a_zip_s_members_are_supplements(self):
        pdf = SourceFile(
            "paper.pdf",
            "paper.pdf",
            "supplied",
            "application/pdf",
            pdf_bytes(["APOE rs429358 at 19:44908684"]),
        )
        tables = SourceFile(
            "tables.zip",
            "tables.zip",
            "supplied",
            "application/zip",
            zipped({"S1.tsv": "rs4680\tCOMT \u2013 22\n".encode(), "S2.bin": b"\x00"}),
        )

        texts = document_texts([pdf, tables])

        self.assertEqual(
            [(d.document_id, d.source, d.name, d.text) for d in texts.documents],
            [
                ("main", 0, "paper.pdf", "## Page 1\nAPOE rs429358 at 19:44908684"),
                ("supp:S1.tsv", 1, "S1.tsv", "rs4680\tCOMT - 22\n"),
            ],
        )
        self.assertEqual(
            [(r.source, r.name, r.reason) for r in texts.readings],
            [
                (0, "paper.pdf", ""),
                (1, "S1.tsv", ""),
                (1, "S2.bin", "no reader for .bin"),
            ],
        )

    def test_a_file_with_no_reader_gives_no_document_and_its_reason(self):
        scan = SourceFile(
            "scan.pdf", "scan.pdf", "supplied", "application/pdf", b"%PDF-1.4"
        )

        texts = document_texts([scan])

        self.assertEqual(texts.documents, ())
        self.assertEqual(
            [(r.name, r.text, r.reason) for r in texts.readings],
            [("scan.pdf", None, "no reader")],
        )

    def test_an_article_that_is_not_jats_is_unread(self):
        article = SourceFile(
            "a.xml", "https://h/a.xml", "article", "text/xml", b"<html"
        )

        texts = document_texts([article])

        self.assertEqual(texts.documents, ())
        self.assertEqual(texts.readings[0].reason, "not JATS")

    def test_a_malformed_supported_file_is_unread_with_its_parser_s_error(self):
        """#420 SP4: a .docx whose XML does not parse is unread, alone or in a zip."""
        broken = zipped({"word/document.xml": b"<w:document><unclosed"})
        alone = SourceFile("paper.docx", "paper.docx", "supplied", "", broken)
        inside = SourceFile(
            "s.zip", "s.zip", "supplied", "application/zip", zipped({"S1.docx": broken})
        )

        self.assertEqual(
            [(r.name, r.reason) for r in document_texts([alone]).readings],
            [("paper.docx", "unreadable: ParseError")],
        )
        self.assertEqual(
            [(r.name, r.reason) for r in document_texts([inside]).readings],
            [("S1.docx", "no reader for .docx (ParseError)")],
        )

    def test_a_workbook_cell_naming_a_missing_shared_string_is_unread(self):
        """#420 SP4 (r2): an .xlsx whose cell names shared string 9 of one is unread."""
        broken = zipped(
            {
                "xl/workbook.xml": b'<workbook xmlns="m"><sheets><sheet name="S1" '
                b'sheetId="1" r:id="rId1" xmlns:r="r"/></sheets></workbook>',
                "xl/_rels/workbook.xml.rels": b'<Relationships xmlns="p"><Relationship '
                b'Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
                "xl/sharedStrings.xml": b'<sst xmlns="m"><si><t>rs1</t></si></sst>',
                "xl/worksheets/sheet1.xml": b'<worksheet xmlns="m"><sheetData><row r="1">'
                b'<c r="A1" t="s"><v>9</v></c></row></sheetData></worksheet>',
            }
        )
        book = SourceFile("t.xlsx", "t.xlsx", "supplied", "", broken)

        (reading,) = document_texts([book]).readings

        self.assertEqual(reading.text, None)
        self.assertTrue(reading.reason.startswith("unreadable: "), reading.reason)

    def test_text_that_is_only_whitespace_is_no_text(self):
        """#420 SP5: a readable file must write some text."""
        blank = SourceFile("blank.txt", "blank.txt", "supplied", "text/plain", b" \n\t")

        texts = document_texts([blank])

        self.assertEqual(texts.documents, ())
        self.assertEqual(texts.readings[0].reason, "no text")


if __name__ == "__main__":
    unittest.main()
