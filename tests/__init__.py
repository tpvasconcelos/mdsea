#!/usr/local/bin/python
# coding: utf-8
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

del os, sys, path

# noinspection PyUnresolvedReferences,PyUnresolvedReferences
import mdsea
