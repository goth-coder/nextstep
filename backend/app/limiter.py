"""
Shared Flask-Limiter instance — initialised in create_app() via init_app().

Import `limiter` here to apply per-route limits with @limiter.limit().
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per hour", "30 per minute"],
    storage_uri="memory://",
)
