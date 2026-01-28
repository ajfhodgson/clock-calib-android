# Define paths
$zipPath = "C:\Users\andre\Downloads\clock-calibrator-apk.zip"
$extractPath = "C:\Users\andre\Downloads\"
$apkFile = "clockcalib-0.1-arm64-v8a_armeabi-v7a-debug.apk"
$localApkPath = Join-Path $extractPath $apkFile
$adbPath = "C:\Program Files\platform-tools\adb.exe"
$packageName = "org.andrew.clock.clockcalib"

# Step 1: Extract APK from ZIP
Write-Host "Extracting APK from ZIP..."
try {
    Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
    Write-Host "Extraction successful"
    
    # Step 2: Delete ZIP file if extraction was successful
    if (Test-Path $localApkPath) {
        Write-Host "Deleting ZIP file..."
        Remove-Item $zipPath -Force
        Write-Host "ZIP file deleted"
    } else {
        Write-Host "ERROR: APK file not found after extraction"
        exit 1
    }
} catch {
    Write-Host "ERROR: Failed to extract ZIP file: $_"
    exit 1
}

# Step 3: Uninstall old app from phone
Write-Host "Uninstalling old app from phone..."
$uninstallResult = & $adbPath uninstall $packageName 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Old app uninstalled successfully"
} else {
    Write-Host "App not found or already uninstalled (this is OK for first install)"
}

# Step 4: Install new APK on phone
Write-Host "Installing new APK on phone..."
$installResult = & $adbPath install $localApkPath 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Installation successful"
    
    # Step 5: Delete local APK file if installation was successful
    Write-Host "Deleting local APK file..."
    Remove-Item $localApkPath -Force
    Write-Host "Local APK file deleted"
    Write-Host "`nAll operations completed successfully!"
} else {
    Write-Host "ERROR: Failed to install APK on phone"
    Write-Host $installResult
    exit 1
}