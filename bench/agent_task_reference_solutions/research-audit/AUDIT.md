# Repository audit

Three modules, one finding each. No source file was modified.

## src/auth.py

`SESSION_TTL_SECONDS` (src/auth.py:9) is one year. A token that leaks stays
valid for a year and there is no revocation list, so an operator cannot expire
it. Every other part of `verify_token` is sound: the comparison is
constant-time. Severity: high.

## src/storage.py

`load_records` catches every exception and returns an empty list
(src/storage.py:13). A permission error, a missing file and a corrupt line are
therefore indistinguishable from an empty store, and the caller silently
receives "no records". Severity: medium.

## src/api.py

`create_record` (src/api.py:13) appends whatever payload it is handed. Nothing
checks the shape, the key set or the size, so a caller can write arbitrary
records into the store. Severity: medium.
