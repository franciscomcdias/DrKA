#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from drka import DATA_DIR
from drka.tokenizers import SpacyTokenizer
from . import config
from . import data
from . import utils
from . import vector
from .model import DocReader
from .predictor import Predictor

DEFAULTS = {
    'tokenizer': SpacyTokenizer,
    'model': os.path.join(DATA_DIR, 'reader/single.mdl'),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value
