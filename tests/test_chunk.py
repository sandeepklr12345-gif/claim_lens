from claimlens.rag.chunk import chunk_pages, words_in

SAMPLE = [
    "SECTION C. EXCLUSIONS\n"
    "4.1 Waiting Periods\n"
    "Pre-existing diseases are covered only after 36 months of continuous coverage. "
    + "Filler sentence about the policy. " * 30
    + "\n4.2 Permanent Exclusions\n"
    "(1) Cosmetic surgery is excluded unless required after an accident.\n"
    "(2) Dental treatment is excluded except for accidents.\n",
    "Page two continues.\n4.3 Claims\nThe insured shall notify the company within 48 hours.\n",
]


def test_section_paths_and_pages():
    chunks = chunk_pages("doc", SAMPLE, min_words=5, max_words=300)
    paths = {c.section_path for c in chunks}
    assert any(p.startswith("SECTION C. EXCLUSIONS > 4.1 Waiting Periods") for p in paths)
    cosmetic = next(c for c in chunks if "Cosmetic" in c.text)
    assert "4.2 Permanent Exclusions" in cosmetic.section_path
    assert "(1)" in cosmetic.section_path
    assert cosmetic.page == 1
    claims = next(c for c in chunks if "notify the company" in c.text)
    assert claims.page == 2 and "4.3 Claims" in claims.section_path


def test_chunk_size_limits():
    chunks = chunk_pages("doc", SAMPLE, min_words=5, max_words=40)
    assert all(words_in(c.text) <= 40 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_repeated_header_removed():
    pages = [
        f"Company Ltd. UIN: X123\n1.{i} Heading {i}\nSome clause text number {i}."
        for i in range(1, 6)
    ]
    chunks = chunk_pages("doc", pages, min_words=1, max_words=300)
    assert all("UIN" not in c.text for c in chunks)


def test_bare_integer_lines_are_not_headings():
    pages = ["2.1 Exclusions\n12. Arthritis and gout\n13. Cataract surgery"]
    chunks = chunk_pages("doc", pages, min_words=1, max_words=300)
    assert all(c.section_path.startswith("2.1 Exclusions") for c in chunks)
