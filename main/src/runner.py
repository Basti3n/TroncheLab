import argparse
import logging
from tkinter import Tk

from main.src.logger.logger_configuration import setup_logging_configuration
from main.src.application import Application

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='TroncheLab App')
    parser.add_argument('--log_level', default='INFO', help='INFO, DEBUG, WARN or ERROR')
    args = parser.parse_args()

    log_level = args.log_level

    setup_logging_configuration(log_level)

    logger.info('** App running  **')
    logger.debug('Debug Working !')

    root = Tk()
    app = Application(master=root, width=300, height=400)
    app.mainloop()


