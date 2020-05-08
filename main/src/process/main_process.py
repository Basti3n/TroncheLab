from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)


@dataclass
class MainProcess:

    @classmethod
    def run(cls) -> None:
        logger.info("Main process started")

        logger.info("Main process ended")



