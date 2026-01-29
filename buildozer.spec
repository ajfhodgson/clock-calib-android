[app]
title = Clock Calibrator
package.name = clockcalib
package.domain = org.andrew.clock
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

# --- Requirements ---
requirements = python3, kivy==2.3.0, numpy, pyjnius==1.6.0, android

orientation = portrait
fullscreen = 0

# --- Android Configuration ---
android.permissions = RECORD_AUDIO
android.api = 33
android.minapi = 21
android.sdk = 33
android.buildtools = 33.0.0
android.accept_sdk_license = True

# --- THE SPEED SECRET: Use GitHub Runner's built-in tools ---
# This prevents Buildozer from trying to download these into a cache that keeps breaking.
android.sdk_path = /usr/local/lib/android/sdk
android.ndk_path = /usr/local/lib/android/sdk/ndk/27.3.13750724
android.skip_update = True

[buildozer]
log_level = 2
warn_on_root = 1