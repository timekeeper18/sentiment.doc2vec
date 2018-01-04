#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Jun 1, 2016

@author: timekeeper
"""

import os
import sys

MODEL_PATH = "{}models/".format(os.path.dirname(os.path.realpath(__file__))[0:-3])
DATA_PATH = "{}imdb/".format(os.path.dirname(os.path.realpath(__file__))[0:-3])

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "verbose": {
            "format": "[%(asctime)s] %(levelname)s in '%(module)s' at line %(lineno)d: %(message)s "
        },
        "simple": {
            "format": "[%(asctime)s] %(levelname)s %(message)s"
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "formatter": "verbose",
            "class": "logging.handlers.RotatingFileHandler",
            "filename":
                os.path.dirname(os.path.realpath(__file__))[0:-3] + "log/{}.log".format(format(sys.argv[0].split("/")[-1][0:-3])),
            "maxBytes": 1048576,
            "backupCount": 10
        },
        "console": {
            "level": "INFO",
            "formatter": "verbose",
            "class": "logging.StreamHandler"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        },
    }
}

if __name__ == '__main__':
    pass
