"""Flask web application for full Late-Fusion inference."""


def create_app(*args, **kwargs):
    """Import the application factory lazily to avoid loading the model twice."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
