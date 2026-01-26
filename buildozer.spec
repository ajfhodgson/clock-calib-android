[app]
title = Clock Calibrator
package.name = clockcalib
package.domain = org.andrew.clock
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

# These are the critical requirements
requirements = python3, kivy, numpy

orientation = portrait
fullscreen = 0

# Android specific permissions
android.permissions = RECORD_AUDIO, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

# (Optionally) Specify the entry point if not main.py
# source.main = main.py

[buildozer]
log_level = 2
warn_on_root = 1