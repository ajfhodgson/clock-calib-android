[app]
title = Clock Calibrator
package.name = clockcalib
package.domain = org.andrew.clock
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

# Requirements
requirements = python3, kivy, numpy

orientation = portrait
fullscreen = 0

# Android specific permissions
android.permissions = RECORD_AUDIO, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

# --- ADD THESE LINES HERE ---
android.api = 34
android.minapi = 21
android.sdk = 34
android.buildtools = 34.0.0
# ----------------------------

[buildozer]
log_level = 2
warn_on_root = 1