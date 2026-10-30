import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.parse
import zipfile
from email.message import Message
from pathlib import Path
from unittest import mock

from bio_neb.paper_reading.contracts import PaperDocument
from bio_neb.paper_reading.reader import grep, rows
from bio_neb.providers.paper_sources import PaperDownload, PaperSourceAdapter
from bio_neb.target_request import request_document
from bio_neb.target_request.paper_text import read as read_values
from tests.paper_reading_support import (
    document,
    finding,
    kept_answer,
    reading_of,
    value,
)
from tools.paper_retrieval.__main__ import _save_documents, load_paper, main
from tools.paper_retrieval.tools import RetrievalBundle, retrieve

# The one request writer's wording beside the paper's line in a Run's /3 request file.
REQUEST_WRITERS = (
    request_document.request_items,
    request_document._unplaced,
    request_document._candidate,
    request_document._text,
)

DOI = "10.1371/journal.pgen.1006651"


REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def europe_pmc_search_url(query: str) -> str:
    return (
        f"{REST}/search?query={urllib.parse.quote_plus(query)}"
        "&format=json&resultType=lite&pageSize=25&cursorMark=%2A"
    )


def europe_pmc_fulltext_url(pmcid: str) -> str:
    return f"{REST}/{pmcid}/fullTextXML"


def pmc_open_data_listing_url(pmcid: str) -> str:
    return f"https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix={pmcid}."


DOG_PMCID = "PMC5352138"
EPMC_ZIP = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5352138/supplementaryFiles"
)
BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
DOG_JATS = (
    '<article xmlns:xlink="http://www.w3.org/1999/xlink"><body><p>Main text.</p></body>'
    "<back><supplementary-material><label>S1 Table</label>"
    '<media xlink:href="pgen.1006651.s001.csv"/></supplementary-material>'
    "<supplementary-material><label>S2 Fig</label>"
    '<media xlink:href="pgen.1006651.s002.pdf"/></supplementary-material></back></article>'
)
DOG_SITES = b"chrom,pos\n10,68587027\n"
DOG_FIGURE = b"%PDF-1.4 figure"


def zipped(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def bucket_listing(pmcid: str, keys: list[str]) -> bytes:
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<Prefix>{pmcid}.</Prefix>{contents}</ListBucketResult>"
    ).encode()


def dog_responses() -> dict[str, tuple[int, bytes, str]]:
    """Recorded shapes of each host's answer for the dog paper; the contact run checks them."""

    generic = "binary/octet-stream"
    return {
        europe_pmc_search_url(f'DOI:"{DOI}"'): (
            200,
            json.dumps(
                {
                    "resultList": {
                        "result": [
                            {"pmcid": DOG_PMCID, "doi": DOI, "isOpenAccess": "Y"}
                        ]
                    }
                }
            ).encode(),
            "application/json",
        ),
        EPMC_ZIP: (
            200,
            zipped(
                {
                    "pgen.1006651.s001.csv": DOG_SITES,
                    "pgen.1006651.s002.pdf": DOG_FIGURE,
                }
            ),
            "application/zip",
        ),
        pmc_open_data_listing_url(DOG_PMCID): (
            200,
            bucket_listing(
                DOG_PMCID,
                [
                    "PMC5352138.1/PMC5352138.1.xml",
                    "PMC5352138.1/pgen.1006651.g001.jpg",
                    "PMC5352138.1/pgen.1006651.s001.csv",
                    "PMC5352138.1/pgen.1006651.s002.pdf",
                ],
            ),
            "application/xml",
        ),
        f"{BUCKET}/PMC5352138.1/PMC5352138.1.xml": (200, DOG_JATS.encode(), generic),
        f"{BUCKET}/PMC5352138.1/pgen.1006651.s001.csv": (200, DOG_SITES, generic),
        f"{BUCKET}/PMC5352138.1/pgen.1006651.s002.pdf": (200, DOG_FIGURE, generic),
    }


def hosts(
    responses: dict[str, tuple[int, bytes, str]],
) -> tuple[PaperSourceAdapter, list[str]]:
    """The product's paper adapter over recorded responses; unknown URLs answer 404."""

    asked: list[str] = []

    def opener(request, timeout):
        url = request.full_url
        asked.append(url)
        status, body, kind = responses.get(url, (404, b"", "text/plain"))
        headers = Message()
        headers["Content-Type"] = kind
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "", headers, io.BytesIO(body))
        response = mock.MagicMock(status=status, headers=headers)
        response.__enter__.return_value = response
        response.read.side_effect = lambda size=-1: body if size < 0 else body[:size]
        return response

    return PaperSourceAdapter(opener=opener), asked


DOG_ARTICLE = (
    '<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta><title-group>'
    "<article-title>A de novo variant in the ASPRV1 gene in a dog with ichthyosis</article-title>"
    "</title-group></article-meta></front><body><sec><title>Methods</title>"
    "<p>We used the dog CanFam 3.1 reference genome assembly for all analyses.</p></sec>"
    "<sec><title>Results</title><p>This de novo variant was a missense variant, c.1052T&gt;C, "
    "located in the ASPRV1 gene.</p></sec></body>"
    "<back><supplementary-material><label>S1 Table</label>"
    '<media xlink:href="pgen.1006651.s001.csv"/></supplementary-material>'
    "<supplementary-material><label>S2 Fig</label>"
    '<media xlink:href="pgen.1006651.s002.pdf"/></supplementary-material></back></article>'
)
DOG_TABLE = b"Chromosome,Position CanFam3.1,Ref,Alt,Gene\n10,68587027,A,G,ASPRV1\n"


def dog_article_responses() -> dict[str, tuple[int, bytes, str]]:
    responses = dog_responses()
    generic = "binary/octet-stream"
    responses[f"{BUCKET}/PMC5352138.1/PMC5352138.1.xml"] = (
        200,
        DOG_ARTICLE.encode(),
        generic,
    )
    responses[f"{BUCKET}/PMC5352138.1/pgen.1006651.s001.csv"] = (
        200,
        DOG_TABLE,
        generic,
    )
    responses[EPMC_ZIP] = (
        200,
        zipped(
            {"pgen.1006651.s001.csv": DOG_TABLE, "pgen.1006651.s002.pdf": DOG_FIGURE}
        ),
        "application/zip",
    )
    return responses


def routes(bundle) -> list[str]:
    """The routes the paper's documents came by, in order."""
    return list(dict.fromkeys(d.route for d in bundle.download.documents))


class DirectRetrievalTests(unittest.TestCase):
    def test_a_pmc_paper_comes_from_the_bucket_with_its_candidates(self):
        adapter, asked = hosts(dog_article_responses())

        bundle = retrieve(f"doi:{DOI}", adapter)

        # The bucket delivered the article and every file it names: Europe PMC's zip is not asked.
        self.assertEqual(
            asked,
            [
                europe_pmc_search_url(f'DOI:"{DOI}"'),
                pmc_open_data_listing_url(DOG_PMCID),
                f"{BUCKET}/PMC5352138.1/PMC5352138.1.xml",
                f"{BUCKET}/PMC5352138.1/pgen.1006651.s001.csv",
                f"{BUCKET}/PMC5352138.1/pgen.1006651.s002.pdf",
            ],
        )
        self.assertEqual(adapter.contact_count, len(asked))
        self.assertEqual(bundle.main.url, f"{BUCKET}/PMC5352138.1/PMC5352138.1.xml")
        self.assertEqual(
            [(s.document_id, s.url) for s in bundle.supplementary],
            [
                (
                    "supp:pgen.1006651.s001.csv",
                    f"{BUCKET}/PMC5352138.1/pgen.1006651.s001.csv",
                )
            ],
        )
        self.assertEqual(routes(bundle), ["pmc_open_data"])
        # The article states its version and the table its row; the model reads them.
        self.assertIn("CanFam 3.1", read_values(bundle.main.text).versions)
        [table] = bundle.supplementary
        self.assertIn("10\t68587027\tA\tG\tASPRV1", table.text.splitlines())

    def test_without_the_bucket_europe_pmc_gives_the_article_and_the_zip(self):
        responses = dog_article_responses()
        responses[pmc_open_data_listing_url(DOG_PMCID)] = (
            200,
            bucket_listing(DOG_PMCID, []),
            "application/xml",
        )
        responses[europe_pmc_fulltext_url(DOG_PMCID)] = (
            200,
            DOG_ARTICLE.encode(),
            "application/xml",
        )
        adapter, asked = hosts(responses)

        bundle = retrieve(f"doi:{DOI}", adapter)

        self.assertEqual(
            asked,
            [
                europe_pmc_search_url(f'DOI:"{DOI}"'),
                pmc_open_data_listing_url(DOG_PMCID),
                europe_pmc_fulltext_url(DOG_PMCID),
                EPMC_ZIP,
            ],
        )
        self.assertEqual(bundle.main.url, europe_pmc_fulltext_url(DOG_PMCID))
        self.assertEqual(
            [s.document_id for s in bundle.supplementary],
            ["supp:pgen.1006651.s001.csv"],
        )
        self.assertEqual(
            routes(bundle), ["europe_pmc_full_text", "europe_pmc_supplementary_files"]
        )
        self.assertIn(68587027, written_positions(bundle.supplementary))

    def archive_only(self, members: dict[str, bytes]):
        """The dog paper with no bucket copy: the article from Europe PMC and the given
        members as its supplement zip."""
        responses = dog_article_responses()
        responses[pmc_open_data_listing_url(DOG_PMCID)] = (
            200,
            bucket_listing(DOG_PMCID, []),
            "application/xml",
        )
        responses[europe_pmc_fulltext_url(DOG_PMCID)] = (
            200,
            DOG_ARTICLE.encode(),
            "application/xml",
        )
        responses[EPMC_ZIP] = (200, zipped(members), "application/zip")
        adapter, _ = hosts(responses)
        return retrieve(f"doi:{DOI}", adapter)

    def test_two_archive_members_with_one_file_name_stay_two_documents(self):
        # Every distinct member keeps its own identity, whatever its path, and is saved
        # under it.
        first = b"chrom\tpos\tref\talt\n1\t123456\tA\tG\n"
        second = b"chrom\tpos\tref\talt\n2\t987654\tC\tT\n"
        for members, ids in (
            (
                {"S1/table.tsv": first, "S2/table.tsv": second},
                ["supp:table.tsv", "supp:S2/table.tsv"],
            ),
            (
                {"S1/table.tsv": first, "table.tsv": second},
                ["supp:table.tsv", "supp:table.tsv#2"],
            ),
        ):
            with self.subTest(members=list(members)):
                bundle = self.archive_only(members)

                self.assertEqual([s.document_id for s in bundle.supplementary], ids)
                self.assertLessEqual(
                    {123456, 987654}, written_positions(bundle.supplementary)
                )
                with tempfile.TemporaryDirectory() as directory:
                    _save_documents(
                        [bundle.main, *bundle.supplementary], Path(directory)
                    )
                    self.assertEqual(
                        Path(directory, f"{ids[1]}.txt").read_text(encoding="utf-8"),
                        second.decode(),
                    )

    def test_a_later_route_keeps_a_distinct_file_of_a_name_already_given(self):
        # A later route's file is the same source only when its name and its text match a
        # file already given; another table of that name stays.
        responses = dog_article_responses()
        responses[f"{BUCKET}/PMC5352138.1/PMC5352138.1.xml"] = (
            200,
            DOG_ARTICLE.replace("pgen.1006651.s002.pdf", "missing.tsv").encode(),
            "binary/octet-stream",
        )
        other = b"chrom\tpos\tref\talt\n2\t987654\tC\tT\n"
        responses[EPMC_ZIP] = (
            200,
            zipped(
                {
                    "S1/pgen.1006651.s001.csv": DOG_TABLE,
                    "S2/pgen.1006651.s001.csv": other,
                }
            ),
            "application/zip",
        )
        adapter, _ = hosts(responses)

        bundle = retrieve(f"doi:{DOI}", adapter)

        self.assertEqual(
            routes(bundle), ["pmc_open_data", "europe_pmc_supplementary_files"]
        )
        self.assertEqual(
            [(s.document_id, s.url) for s in bundle.supplementary],
            [
                (
                    "supp:pgen.1006651.s001.csv",
                    f"{BUCKET}/PMC5352138.1/pgen.1006651.s001.csv",
                ),
                (
                    "supp:S2/pgen.1006651.s001.csv",
                    f"{EPMC_ZIP}#S2/pgen.1006651.s001.csv",
                ),
            ],
        )
        self.assertLessEqual(
            {68587027, 987654}, written_positions(bundle.supplementary)
        )

    def test_the_bucket_files_are_not_capped(self):
        names = [f"pgen.1006651.s{index:03d}.csv" for index in range(1, 26)]
        article = DOG_ARTICLE.replace(
            '<media xlink:href="pgen.1006651.s001.csv"/>',
            "".join(f'<media xlink:href="{name}"/>' for name in names),
        )
        responses = dog_article_responses()
        generic = "binary/octet-stream"
        responses[pmc_open_data_listing_url(DOG_PMCID)] = (
            200,
            bucket_listing(
                DOG_PMCID,
                ["PMC5352138.1/PMC5352138.1.xml", "PMC5352138.1/pgen.1006651.s002.pdf"]
                + [f"PMC5352138.1/{name}" for name in names],
            ),
            "application/xml",
        )
        responses[f"{BUCKET}/PMC5352138.1/PMC5352138.1.xml"] = (
            200,
            article.encode(),
            generic,
        )
        for index, name in enumerate(names):
            row = f"chrom,pos,ref,alt\n7,{100000 + index},C,T\n".encode()
            responses[f"{BUCKET}/PMC5352138.1/{name}"] = (200, row, generic)
        adapter, _ = hosts(responses)

        # Outside a Run no Tool count applies: every named file is downloaded.
        bundle = retrieve(f"doi:{DOI}", adapter)

        self.assertEqual(len(bundle.supplementary), 25)
        self.assertLessEqual(
            {100000 + index for index in range(25)},
            written_positions(bundle.supplementary),
        )


def written_positions(documents) -> set[int]:
    """Every position the general parser reads in the documents' rows (two-column and
    wider comma rows are read as the tab rows the readers write)."""

    return {
        position
        for document in documents
        if isinstance(document, PaperDocument)
        for line in document.text.splitlines()
        for position in read_values(line.replace(",", "\t")).positions
    }


class CommandLineTests(unittest.TestCase):
    def test_retrieve_writes_each_document_under_the_id_its_listing_names(self):
        # Review r9 (Standards S10, Spec S11): paper.json and the files use one id set.
        renamed, clean = (
            document(name, f"## Results\n{gene} c.100A>G\n", "supplementary")
            for name, gene in (
                ("supp:table\x07a.tsv", "GENE1"),
                ("supp:table a.tsv", "GENE2"),
            )
        )
        bundle = RetrievalBundle(
            reference="doi:10.1234/test",
            identity=None,
            download=PaperDownload(reference="doi:10.1234/test"),
            via="recorded",
            main=None,
            supplementary=[renamed, clean],
            files=[],
        )
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with (
                mock.patch(
                    "tools.paper_retrieval.__main__.retrieve", return_value=bundle
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = main(["retrieve", "doi:10.1234/test", "--out", str(out)])
            reopened = load_paper(out)

        self.assertEqual(code, 0)
        self.assertEqual(
            [(d.document_id, d.text) for d in reopened.documents.values()],
            [("supp:table a.tsv#2", renamed.text), ("supp:table a.tsv", clean.text)],
        )

    def run_retrieve(self, out: Path, *options: str, answers=None) -> int:
        """``retrieve``; with ``answers``, the product's adapter answers from them."""

        adapter, _ = hosts(answers or {})
        with (
            mock.patch(
                "tools.paper_retrieval.__main__.PaperSourceAdapter",
                side_effect=lambda opener=None: adapter
                if opener is None
                else PaperSourceAdapter(opener=opener),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return main(["retrieve", f"doi:{DOI}", "--out", str(out), *options])

    def test_retrieve_writes_the_paper_its_files_and_every_request(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root, "out")
            code = self.run_retrieve(out, answers=dog_article_responses())
            recorded = json.loads((out / "paper.json").read_text())
            downloaded = json.loads((out / "download.json").read_text())
            table = (out / "files" / "pgen.1006651.s001.csv").read_bytes()

        self.assertEqual(code, 0)
        # Each GET is the product's: one GET to a named host, no redirect, no search.
        self.assertEqual((downloaded["sent"], downloaded["stop"]), (5, None))
        self.assertEqual(table, DOG_TABLE)
        self.assertEqual(
            [
                (d["name"], d["route"], d["bytes"], d["sha256"], d["md5"])
                for d in downloaded["documents"]
            ][1],
            (
                "pgen.1006651.s001.csv",
                "pmc_open_data",
                len(DOG_TABLE),
                hashlib.sha256(DOG_TABLE).hexdigest(),
                hashlib.md5(DOG_TABLE).hexdigest(),
            ),
        )
        self.assertEqual(recorded["identity"]["pmcid"], DOG_PMCID)
        self.assertEqual(
            [d["document_id"] for d in recorded["documents"]],
            ["main", "supp:pgen.1006651.s001.csv"],
        )

    def test_retrieve_replays_a_raw_capture_without_contact(self):
        answers = dog_article_responses()
        with tempfile.TemporaryDirectory() as root:
            raw, live, replay = (
                Path(root, "raw"),
                Path(root, "live"),
                Path(root, "replay"),
            )
            raw.mkdir()
            # A raw capture as lane P kept it: an index line per URL, each body by digest.
            with (raw / "index.jsonl").open("w") as index:
                for url, (status, body, kind) in answers.items():
                    digest = hashlib.sha256(body).hexdigest()
                    (raw / digest).write_bytes(body)
                    index.write(
                        json.dumps(
                            {
                                "url": url,
                                "final_url": url,
                                "status": status,
                                "content_type": kind,
                                "sha256": digest,
                            }
                        )
                        + "\n"
                    )
            self.run_retrieve(live, answers=answers)
            code = self.run_retrieve(replay, "--replay", str(raw))
            first = json.loads((live / "paper.json").read_text())
            again = json.loads((replay / "paper.json").read_text())
            texts = [
                (folder / "documents" / f"{d['document_id']}.txt").read_bytes()
                for folder, recorded in ((live, first), (replay, again))
                for d in recorded["documents"]
            ]
            bundle = json.loads((replay / "bundle.json").read_text())
            downloaded = json.loads((replay / "download.json").read_text())

        self.assertEqual(code, 0)
        # The same bytes give the same documents; the replay says it is a replay.
        half = len(texts) // 2
        self.assertEqual(texts[:half], texts[half:])
        self.assertEqual(
            [(d["document_id"], d["characters"]) for d in again["documents"]],
            [(d["document_id"], d["characters"]) for d in first["documents"]],
        )
        self.assertEqual({d["via"] for d in again["documents"]}, {"recorded"})
        self.assertEqual(bundle["replayed_from"], str(raw))
        self.assertEqual((downloaded["sent"], downloaded["stop"]), (5, None))

    def test_a_stopped_download_writes_what_it_had_and_fails(self):
        answers = dog_article_responses()
        answers[f"{BUCKET}/PMC5352138.1/pgen.1006651.s001.csv"] = (
            403,
            b"No",
            "text/plain",
        )
        with tempfile.TemporaryDirectory() as root:
            out = Path(root, "out")
            code = self.run_retrieve(out, answers=answers)
            downloaded = json.loads((out / "download.json").read_text())

        self.assertEqual(code, 1)
        self.assertEqual(downloaded["sent"], 4)
        self.assertEqual(downloaded["stop"]["host"], "pmc-oa-opendata.s3.amazonaws.com")
        self.assertEqual([d["role"] for d in downloaded["documents"]], ["article"])

    def retrieved_dog(self, root: Path) -> Path:
        """The dog paper as ``retrieve`` writes it: paper.json and its documents."""

        out = root / "paper"
        self.run_retrieve(out, answers=dog_article_responses())
        return out

    def command(self, *argv: str) -> tuple[int, str]:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = main(list(argv))
        return code, printed.getvalue()

    def test_the_reading_commands_read_a_retrieved_paper(self):
        with tempfile.TemporaryDirectory() as root:
            paper = str(self.retrieved_dog(Path(root)))

            outlined = self.command("outline", "--paper", paper)
            found = self.command("grep", "--paper", paper, "68587027")
            row = self.command(
                "rows", "--paper", paper, "--table", "t1", "--where", "Gene=ASPRV1"
            )
            text = self.command("read", "--paper", paper, "main", "--length", "40")
            listed = self.command("lookup", "--paper", paper, "10:68587027", "rs1")
            refused = self.command("read", "--paper", paper, "s99")

        csv = "supp:pgen.1006651.s001.csv"
        self.assertEqual(outlined[0], 0)
        [table] = [
            t
            for d in json.loads(outlined[1])["documents"]
            if d["document_id"] == csv
            for t in d["tables"]
        ]
        self.assertEqual(
            table["columns"], ["Chromosome", "Position CanFam3.1", "Ref", "Alt", "Gene"]
        )
        self.assertEqual(json.loads(found[1])["matches"], {"main": 0, csv: 1})
        [cells] = [r["cells"] for r in json.loads(row[1])["rows"]]
        self.assertEqual(cells["Position CanFam3.1"], "68587027")
        self.assertEqual(
            json.loads(text[1])["passages"],
            [{"span": "main:p1", "text": "# A de novo variant in the ASPRV1 gene i"}],
        )
        self.assertEqual(
            [(f["id"], f["found"]) for f in json.loads(listed[1])],
            [("10:68587027", True), ("rs1", False)],
        )
        self.assertEqual(
            (refused[0], json.loads(refused[1])["code"]), (1, "unknown_view")
        )

    def test_verify_checks_an_answer_file_against_the_paper(self):
        with tempfile.TemporaryDirectory() as root:
            folder = self.retrieved_dog(Path(root))
            paper = load_paper(folder)
            [methods] = grep(paper, "CanFam 3.1").hits
            [row] = rows(paper, "t1", where={"Gene": "ASPRV1"}).rows
            reading = reading_of(
                finding(
                    value("contig", "10", row.span, column="Chromosome"),
                    value(
                        "position",
                        "68587027",
                        row.span,
                        column="Position CanFam3.1",
                        number=68587027,
                    ),
                    value("ref", "A", row.span, column="Ref"),
                    value("alt", "G", row.span, column="Alt"),
                    value("gene", "ASPRV1", row.span, column="Gene"),
                    reason="The de novo missense variant.",
                ),
                reference={"name": "CanFam 3.1", "span": methods.span, "reason": "R."},
                not_answered=None,
            )
            answer = Path(root, "answer.json")
            filled = kept_answer("What is the finding?", reading, paper)
            self.assertEqual(len(filled.findings), 1)
            answer.write_text(filled.model_dump_json(), encoding="utf-8")
            tampered = Path(root, "tampered.json")
            tampered.write_text(
                # One copied byte of the row changes; the stated value does not.
                filled.model_dump_json().replace("10\\t68587027", "10\\t68587028"),
                encoding="utf-8",
            )
            verified = self.command(
                "verify",
                "--paper",
                str(folder),
                "--answer",
                str(answer),
                "--out",
                str(Path(root, "verify.json")),
            )
            failed = self.command(
                "verify", "--paper", str(folder), "--answer", str(tampered)
            )
            kept = json.loads(Path(root, "verify.json").read_text(encoding="utf-8"))

        self.assertEqual((verified[0], kept["verified"]), (0, True))
        self.assertEqual(failed[0], 1)
        problems = json.loads(failed[1])["problems"]
        # The copied row differs; each kept value is still where the paper writes it.
        self.assertEqual(
            problems, [f"passage_differs:1:{row.document_id}@{row.offset}"]
        )
