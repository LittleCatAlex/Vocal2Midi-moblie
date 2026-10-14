import logging
from .base import Converter, G2PConversionError, G2PGroup, G2PPath, G2PWord, G2PReading

logger = logging.getLogger(__name__)

# Explicitly import all built-in converter modules so their @converter decorators
# execute deterministically across all environments (especially on Android where
# bytecode compilation / zip bundling breaks dynamic .iterdir() with suffix == ".py").
try:
    from . import chinese
except Exception as exc:
    logger.warning(f"Failed to import chinese converter: {exc}")

try:
    from . import dictionary
except Exception as exc:
    logger.warning(f"Failed to import dictionary converter: {exc}")

try:
    from . import japanese
except Exception as exc:
    logger.warning(f"Failed to import japanese converter: {exc}")

try:
    from . import lstm
except Exception as exc:
    logger.warning(f"Failed to import lstm converter: {exc}")

try:
    from . import paradigm
except Exception as exc:
    logger.warning(f"Failed to import paradigm converter: {exc}")

try:
    from . import simple
except Exception as exc:
    logger.warning(f"Failed to import simple converter: {exc}")

try:
    from . import text
except Exception as exc:
    logger.warning(f"Failed to import text converter: {exc}")
