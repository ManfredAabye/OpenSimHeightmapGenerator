@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
echo ==============================================
echo OpenSim Heightmap Generator - EXE Build
echo ==============================================
echo.

set "PY_CMD="
where py >nul 2>&1 && set "PY_CMD=py -3"
if not defined PY_CMD (
    where python >nul 2>&1 && set "PY_CMD=python"
)

if not defined PY_CMD (
    echo "[ERROR] Python wurde nicht gefunden."
    echo "[ERROR] Bitte Python 3 installieren und PATH pruefen."
    goto :fail
)

if not exist ".buildvenv\Scripts\python.exe" (
    echo "[INFO] Erstelle lokale Build-Umgebung (.buildvenv)..."
    %PY_CMD% -m venv .buildvenv
    if errorlevel 1 (
        echo "[ERROR] Virtuelle Umgebung konnte nicht erstellt werden."
        goto :fail
    )
)

call ".buildvenv\Scripts\activate.bat"
if errorlevel 1 (
    echo "[ERROR] Aktivierung der virtuellen Umgebung fehlgeschlagen."
    goto :fail
)

echo "[INFO] Installiere/aktualisiere Build-Abhaengigkeiten..."
python -m pip install --upgrade pip
python -m pip install --upgrade pyinstaller numpy scipy pillow noise matplotlib ttkbootstrap
if errorlevel 1 (
    echo "[ERROR] Paketinstallation fehlgeschlagen."
    goto :fail
)

set "APP_SCRIPT=HeightmapGenerator.py"
if not exist "%APP_SCRIPT%" (
    set "APP_SCRIPT=HeightmapGeneratorgui.py"
)

if not exist "%APP_SCRIPT%" (
    echo "[ERROR] Kein Startskript gefunden (HeightmapGenerator.py / HeightmapGeneratorgui.py)."
    goto :fail
)

if exist "build" rmdir /s /q "build"
if exist "dist\OpenSim-Heightmap-Generator" rmdir /s /q "dist\OpenSim-Heightmap-Generator"
if exist "dist\OpenSim-Heightmap-Generator-OneFile.exe" del /q "dist\OpenSim-Heightmap-Generator-OneFile.exe"

echo "[INFO] Baue Ordner-Build aus %APP_SCRIPT% ..."
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --name "OpenSim-Heightmap-Generator" ^
  --paths "." ^
  --collect-all ttkbootstrap ^
  --hidden-import PIL._tkinter_finder ^
  "%APP_SCRIPT%"

if errorlevel 1 (
        echo "[ERROR] Ordner-Build fehlgeschlagen."
        goto :fail
)

echo.
echo "[INFO] Baue Einzel-EXE (onefile) aus %APP_SCRIPT% ..."
python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name "OpenSim-Heightmap-Generator-OneFile" ^
    --paths "." ^
    --collect-all ttkbootstrap ^
    --hidden-import PIL._tkinter_finder ^
    "%APP_SCRIPT%"

if errorlevel 1 (
        echo "[ERROR] Onefile-Build fehlgeschlagen."
    goto :fail
)

echo.
echo "[OK] Build erfolgreich."
echo "[INFO] Ordner-Build: dist\OpenSim-Heightmap-Generator\OpenSim-Heightmap-Generator.exe"
echo "[INFO] Einzel-EXE : dist\OpenSim-Heightmap-Generator-OneFile.exe"
echo.
goto :end

:fail
echo.
echo "[ERROR] Build wurde mit Fehler beendet."
echo.
pause
exit /b 1

:end
pause
exit /b 0
