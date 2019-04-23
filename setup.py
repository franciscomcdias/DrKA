#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

with open('README.md', errors='ignore') as f:
    readme = f.read()

with open('LICENSE', errors='ignore') as f:
    _license = f.read()

with open('requirements.txt', errors='ignore') as f:
    _reqs = f.read()

setup(
    name='drqa',
    version='0.1.0-fdias.2',
    description='Document Reader Knowledge Assistant',
    long_description=readme,
    license=_license,
    python_requires='>=3.5',
    packages=find_packages(exclude=('data')),
    install_requires=_reqs.strip().split('\n'),
)
