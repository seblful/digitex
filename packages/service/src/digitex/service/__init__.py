"""The composition root of the deployed half.

`digitex.bot` is written against the protocols in `digitex.domain.ports` and
`digitex.db` provides classes that answer to them; neither names the other.
This package is where the two are wired together — the one place that turns a
psycopg pool into the transaction factory the handlers take, and the only place
allowed to name both sides.

It sits above both in the layering contract, which is what lets the inversion
be stated as a rule rather than maintained as a habit: `digitex.bot` may not
import `digitex.db`, `psycopg` or `psycopg_pool` at all.
"""
