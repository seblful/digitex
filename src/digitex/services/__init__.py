"""Services — domain logic extracted out of CLI command bodies.

Each service is a plain class that takes its inputs at construction and exposes
a single ``run`` / ``validate`` / ``count`` method. CLI commands stay tiny:
parse args, instantiate a service, render the result.
"""
