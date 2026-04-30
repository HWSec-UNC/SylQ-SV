import logging
import sys
logger = logging.getLogger('SYLQ_LOGGER')
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)