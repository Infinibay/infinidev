from __future__ import annotations

from argparse import Namespace

import pytest

from bench.build_spanish_embedding_corpus import (
    build,
    iter_django_girls,
    iter_python_docs,
)


def test_python_docs_extracts_translations_with_provenance(tmp_path) -> None:
    po = tmp_path / "library" / "example.po"
    po.parent.mkdir()
    po.write_text(
        '''msgid ""
msgstr "Language: es\\nContent-Type: text/plain; charset=UTF-8\\n"

msgid "Open a file and return its contents."
msgstr "Abre un archivo y retorna su contenido para procesarlo."

#, fuzzy
msgid "Old translation"
msgstr "Esta traducción incompleta no debe incorporarse al corpus."
''',
        encoding="utf-8",
    )

    records = list(iter_python_docs(tmp_path, max_chars=300))

    assert [record["text"] for record in records] == [
        "Abre un archivo y retorna su contenido para procesarlo."
    ]
    assert records[0]["license"] == "PSF-2.0"
    assert records[0]["path"] == "library/example.po"
    assert records[0]["language"] == "es"


def test_django_markdown_keeps_heading_and_code(tmp_path) -> None:
    markdown = tmp_path / "es" / "views" / "README.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text(
        """# Crear una vista

Una vista recibe una petición HTTP y devuelve una respuesta del servidor.

```python
def detail(request):
    return render(request, "detail.html")
```
""",
        encoding="utf-8",
    )

    records = list(iter_django_girls(tmp_path, max_chars=300))

    assert [record["kind"] for record in records] == ["technical_prose", "code"]
    assert all(str(record["text"]).startswith("Crear una vista") for record in records)


def test_sharealike_source_requires_explicit_admission(tmp_path) -> None:
    args = Namespace(
        python_docs=None,
        django_girls=tmp_path,
        include_sharealike=False,
        max_chars=300,
        max_records=None,
    )

    with pytest.raises(SystemExit, match="--include-sharealike"):
        build(args)
