"""Scraping-specific exceptions.

Mapped to HTTP status codes in the router:
  NetworkError   → 502 Bad Gateway   (we couldn't reach the page)
  ExtractionError → 422 Unprocessable Entity (page fetched, but not a product)
"""
from __future__ import annotations


class ScrapingError(Exception):
    """Base class for all scraping errors."""


class NetworkError(ScrapingError):
    """Raised when the target URL cannot be fetched.

    Causes: timeout, DNS failure, non-2xx response, redirect loop.
    """


class ExtractionError(ScrapingError):
    """Raised when the page was fetched but product data cannot be extracted.

    Causes: not a product page, missing name/price, LLM flagged as non-product.
    """
