[app]
title = Clock Calibrator
package.name = clockcalib
package.domain = org.andrew.clock
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

# Requirements
requirements = python3, kivy==2.3.0, numpy, pyjnius==1.6.0, android

orientation = portrait
fullscreen = 0

# Android specific permissions
android.permissions = RECORD_AUDIO

android.api = 33
android.minapi = 21
android.sdk = 33
android.buildtools = 33.0.0

# Add this to skip manual license prompts:
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1