"""Typed errors shared by image feature extractor adapters."""


class ExtractionError(RuntimeError):
    """Base extractor error."""


class ExtractorUnavailableError(ExtractionError):
    """The external extraction engine or dependency is unavailable."""


class ExtractionInputError(ExtractionError):
    """The uploaded image/mask pair is invalid for extraction."""


class ExtractionExecutionError(ExtractionError):
    """The extraction engine started but could not produce valid features."""
