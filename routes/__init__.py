"""
routes/__init__.py
==================
Makes the routes/ directory a Python package and collects all Flask
Blueprints in one place so app.py can register them with a single import.

Usage in app.py:
    from routes import register_blueprints
    register_blueprints(app)

Or individually:
    from routes.health      import health_bp
    from routes.predictions import predictions_bp
    from routes.clusters    import clusters_bp
    from routes.dashboard   import dashboard_bp
"""

from flask import Flask

from routes.health      import health_bp
from routes.predictions import predictions_bp
from routes.clusters    import clusters_bp
from routes.dashboard   import dashboard_bp

# Ordered list used by register_blueprints()
_BLUEPRINTS = [
    (health_bp,      '/api/ai'),
    (predictions_bp, '/api/ai'),
    (clusters_bp,    '/api/ai'),
    (dashboard_bp,   '/api/ai'),
]


def register_blueprints(app: Flask) -> None:
    """
    Register all AI module Blueprints on a Flask application instance.
    Skips any Blueprint that is already registered (safe for re-import).

    Args:
        app: The Flask application to register Blueprints on.
    """
    already_registered = set(app.blueprints.keys())
    for bp, prefix in _BLUEPRINTS:
        if bp.name not in already_registered:
            app.register_blueprint(bp, url_prefix=prefix)


__all__ = [
    "register_blueprints",
    "health_bp",
    "predictions_bp",
    "clusters_bp",
    "dashboard_bp",
]
