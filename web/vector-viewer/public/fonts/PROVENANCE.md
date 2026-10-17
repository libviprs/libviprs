# Glyph files

`Noto Sans Regular/*.pbf` are MapLibre glyph ranges (signed distance field glyphs, 256 code points
per file) taken unchanged from https://github.com/maplibre/demotiles at commit
`601ae60796ceceda2cbd2ed3d2ea92d17a84be4b`, `font/Noto Sans Regular/`. The demotiles repository is
BSD-3-Clause; the Noto Sans typeface is SIL Open Font License 1.1.

Only the ranges a drawing's labels plausibly need are kept: Latin (0-255, 256-511), general
punctuation (8192-8447), letterlike symbols (8448-8703) and mathematical operators (8704-8959, which
holds the diameter sign U+2205 that `%%c` becomes). A label that uses another range is drawn with
the browser's own font instead, and the map logs that it could not load the range.

| file | sha256 |
| --- | --- |
| `0-255.pbf` | `ef1f38a3f1978591e846e9eaddf8a54f7047f546fc6aaed7872cc53151a5de78` |
| `256-511.pbf` | `992557f9e2574cfb9a75e16c7b52832578849772b7c9a6e480d6213fb008fe5d` |
| `8192-8447.pbf` | `f061e5b4e42e5925228534d8822b423a029569263db09f5ce44573600d45295f` |
| `8448-8703.pbf` | `9e333bdeee3c44c429fb7f6911b3c1602a69402d541908832282d5eb3a467003` |
| `8704-8959.pbf` | `5d3487a5bc60e2a68545c4b7aadcd12664559f1d7b6962f760ca98a1b39a5104` |
