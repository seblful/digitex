"""Entry points for the two commands production runs.

`digitex-bot` starts the polling loop; `digitex-db` applies migrations and
loads the corpus. Both resolve `Settings` here, at the boundary, and thread the
result down — nothing deeper reads a file or an environment variable at import.
"""
