[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvDir = Join-Path $repoRoot ".venv-build"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$buildDir = Join-Path $repoRoot "build"
$launcher = Join-Path $buildDir "pyinstaller-entry.py"
$cookieLauncher = Join-Path $buildDir "pyinstaller-cookie-entry.py"
$distDir = Join-Path $repoRoot "dist"
$logoPath = Join-Path $repoRoot "logo.png"
$iconPath = Join-Path $buildDir "gemini-web2api.ico"
$exePath = Join-Path $distDir "gemini-web2api.exe"
$cookieExePath = Join-Path $distDir "gemini-web2api-cookie.exe"

Set-Location $repoRoot

if (-not (Test-Path $venvPython)) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3 -m venv $venvDir
    }
    else {
        $python = Get-Command python -ErrorAction Stop
        & $python.Source -m venv $venvDir
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create build virtual environment."
    }
}

& $venvPython -m pip install --disable-pip-version-check -r requirements.txt "pyinstaller>=6,<7" "pillow>=10,<13"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install build dependencies."
}

$testsPassed = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
    & $venvPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -eq 0) {
        $testsPassed = $true
        break
    }
    if ($attempt -lt 3) {
        Write-Warning "Tests failed on attempt $attempt; retrying the full suite for transient Windows socket resets."
        Start-Sleep -Milliseconds 500
    }
}
if (-not $testsPassed) {
    throw "Tests failed three times; executable was not built."
}

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
& $venvPython -c "from PIL import Image; image=Image.open(r'$logoPath').convert('RGBA').crop((64,180,520,636)).resize((256,256),Image.Resampling.LANCZOS); image.save(r'$iconPath',format='ICO',sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to generate Windows icon from logo.png."
}
@'
import json
import os
import sys

from gemini_web2api.__main__ import main
from gemini_web2api.config import DEFAULT_CONFIG


def create_json_if_missing(path, data):
    try:
        with open(path, "x", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
    except FileExistsError:
        pass
    except OSError as error:
        print(f"Warning: could not create {path}: {error}", file=sys.stderr)


if __name__ == "__main__":
    executable_dir = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
    config_path = os.path.join(executable_dir, "config.json")
    has_config_arg = any(
        arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]
    )
    defaults = ["--tray"]
    if not has_config_arg:
        create_json_if_missing(config_path, DEFAULT_CONFIG)
        if os.path.isfile(config_path):
            defaults.extend(["--config", config_path])
    sys.argv[1:1] = defaults
    main()
'@ | Set-Content -Encoding utf8 $launcher

@'
import json
import os
import sys

from gemini_web2api.__main__ import main
from gemini_web2api.config import DEFAULT_CONFIG


AUTH_TEMPLATE = {
    "cookie": "",
    "sapisid": None,
    "auth_user": None,
    "xsrf_token": None,
    "gemini_bl": None,
}


def create_json_if_missing(path, data):
    try:
        with open(path, "x", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
    except FileExistsError:
        pass
    except OSError as error:
        print(f"Warning: could not create {path}: {error}", file=sys.stderr)


def auth_file_has_credentials(path):
    try:
        with open(path, encoding="utf-8") as file:
            content = file.read().strip()
        if not content.startswith("{"):
            return bool(content)
        cookie = json.loads(content).get("cookie")
        return isinstance(cookie, str) and bool(cookie.strip())
    except (OSError, ValueError, AttributeError):
        return True


def config_has_cookie_path(path):
    try:
        with open(path, encoding="utf-8") as file:
            return bool(json.load(file).get("cookie_file"))
    except (OSError, ValueError, AttributeError):
        return False


if __name__ == "__main__":
    executable_dir = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
    defaults = ["--tray"]
    config_path = os.path.join(executable_dir, "config.json")
    auth_path = os.path.join(executable_dir, "gemini-auth.json")
    legacy_cookie_path = os.path.join(executable_dir, "cookie.txt")
    has_config_arg = any(
        arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]
    )
    if not has_config_arg:
        create_json_if_missing(config_path, DEFAULT_CONFIG)
        if os.path.isfile(config_path):
            defaults.extend(["--config", config_path])
    create_json_if_missing(auth_path, AUTH_TEMPLATE)

    has_cookie_arg = any(
        arg == "--cookie-file" or arg.startswith("--cookie-file=") for arg in sys.argv[1:]
    )
    if not has_cookie_arg:
        if os.path.isfile(auth_path) and auth_file_has_credentials(auth_path):
            defaults.extend(["--cookie-file", auth_path])
        elif config_has_cookie_path(config_path):
            pass
        elif os.path.isfile(legacy_cookie_path):
            defaults.extend(["--cookie-file", legacy_cookie_path])
        elif os.path.isfile(auth_path):
            defaults.extend(["--cookie-file", auth_path])
    sys.argv[1:1] = defaults
    main()
'@ | Set-Content -Encoding utf8 $cookieLauncher

& $venvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --icon $iconPath `
    --name gemini-web2api `
    --paths $repoRoot `
    --distpath $distDir `
    --workpath (Join-Path $buildDir "pyinstaller") `
    --specpath $buildDir `
    $launcher
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

& $venvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --icon $iconPath `
    --name gemini-web2api-cookie `
    --paths $repoRoot `
    --distpath $distDir `
    --workpath (Join-Path $buildDir "pyinstaller-cookie") `
    --specpath $buildDir `
    $cookieLauncher
if ($LASTEXITCODE -ne 0) {
    throw "Cookie executable PyInstaller build failed."
}

$versionCheck = Start-Process -FilePath $exePath -ArgumentList "--version" -Wait -PassThru
$cookieVersionCheck = Start-Process -FilePath $cookieExePath -ArgumentList "--version" -Wait -PassThru
if ($versionCheck.ExitCode -ne 0 -or $cookieVersionCheck.ExitCode -ne 0) {
    throw "Executable smoke check failed."
}

Write-Host "Built $exePath"
Write-Host "Built $cookieExePath"
Write-Host "Tray executable smoke checks passed."
