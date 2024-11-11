@echo off
setlocal enabledelayedexpansion

:: Set console colors
color 0B

:: Set script title
title Configuration Extractor Setup

:: Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Set up logging
set "INSTALL_LOG=%SCRIPT_DIR%install.log"
echo Installation started at %date% %time% > "%INSTALL_LOG%"

:: Function to log messages
call :log "=== Configuration Extractor Setup ==="
call :log "Starting setup process..."

:: Check Python installation
python --version > temp.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [91mPython is not installed or not in PATH[0m
    echo Please install Python 3.8 or higher from https://www.python.org/
    echo After installation, run this script again.
    pause
    exit /b 1
)

set /p PYTHON_VERSION=<temp.txt
del temp.txt

:: Check if we're already in a virtual environment using a simpler approach
python -c "import sys; print('venv' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else '')" > venv_check.txt
set /p VENV_ACTIVE=<venv_check.txt
del venv_check.txt

if "%VENV_ACTIVE%"=="venv" (
    echo [92mAlready in a virtual environment[0m
    call :log "Using existing active virtual environment"
    goto :check_requirements
)

:: Look for PyCharm venv in standard locations
for %%I in (venv .venv) do (
    if exist "%%I\Scripts\activate.bat" (
        echo [92mFound virtual environment: %%I[0m
        call :log "Found virtual environment: %%I"
        call "%%I\Scripts\activate.bat"
        if !ERRORLEVEL! EQU 0 (
            goto :check_requirements
        ) else (
            echo [93mFailed to activate %%I, trying next...[0m
            call :log "Failed to activate %%I"
        )
    )
)

:: If no venv found or activation failed, create new one
echo [93mNo working virtual environment found. Creating new one...[0m
call :log "Creating new virtual environment"
python -m venv .venv
if !ERRORLEVEL! NEQ 0 (
    echo [91mFailed to create virtual environment[0m
    call :log "Failed to create virtual environment"
    pause
    exit /b 1
)

:: Activate the new virtual environment
call ".venv\Scripts\activate.bat"
if !ERRORLEVEL! NEQ 0 (
    echo [91mFailed to activate new virtual environment[0m
    call :log "Failed to activate new virtual environment"
    pause
    exit /b 1
)

:check_requirements
:: Upgrade pip first
python -m pip install --upgrade pip

:: Create a temporary file to check installed packages
python -m pip freeze > temp_requirements.txt

:: Parse requirements.txt and check each package
for /f "tokens=1,* delims==" %%a in ('type requirements.txt ^| findstr /v "#" ^| findstr /v "^$"') do (
    set "pkg=%%a"
    if defined pkg (
        set "pkg=!pkg: =!"
        echo Checking !pkg!...

        :: Skip platform-specific packages that don't match current platform
        echo !pkg! | findstr /C:"platform_system" >nul
        if !ERRORLEVEL! EQU 0 (
            if "!pkg:~-8!"=="Windows'" (
                echo !pkg! | findstr /C:"platform_system == 'Windows'" >nul
                if !ERRORLEVEL! NEQ 0 (
                    continue
                )
                set "pkg=!pkg:;platform_system == 'Windows'=!"
            ) else (
                continue
            )
        )

        findstr /i /c:"!pkg!" temp_requirements.txt >nul
        if !ERRORLEVEL! NEQ 0 (
            echo Installing !pkg!...
            python -m pip install !pkg!
            if !ERRORLEVEL! NEQ 0 (
                echo [91mFailed to install !pkg![0m
                call :log "Failed to install !pkg!"
            )
        )
    )
)

:: Clean up temporary file
del temp_requirements.txt

:: Install missing packages if any
if not "!MISSING_PACKAGES!"=="" (
    echo [93mInstalling missing packages...[0m
    call :log "Installing missing packages: !MISSING_PACKAGES!"
    python -m pip install !MISSING_PACKAGES!
    if !ERRORLEVEL! NEQ 0 (
        echo [91mFailed to install some packages[0m
        call :log "Failed to install packages"
        pause
        exit /b 1
    )
) else (
    echo [92mAll required packages are already installed[0m
    call :log "All required packages are already installed"
)

:: Create necessary directories
echo [92mChecking necessary directories...[0m
for %%D in (temp output logs schema) do (
    if not exist "%%D" (
        echo Creating directory: %%D
        mkdir "%%D"
        call :log "Created directory: %%D"
    )
)

:: Check for .env file
if not exist ".env" (
    call :log "Creating .env file from template..."
    echo [92mCreating .env file...[0m
    (
        echo # OpenAI API Configuration
        echo OPENAI_API_KEY=your-api-key-here
        echo OPENAI_GPT4_MODEL=gpt-4
        echo OPENAI_VISION_MODEL=gpt-4-vision-preview
        echo OPENAI_WHISPER_MODEL=whisper-1
        echo OPENAI_MAX_TOKENS=4096
        echo OPENAI_TEMPERATURE=0.0
        echo OPENAI_TIMEOUT=300
        echo.
        echo # Processing Configuration
        echo PROCESSING_CHUNK_SIZE=2000
        echo PROCESSING_CHUNK_OVERLAP=200
        echo PROCESSING_MAX_RETRIES=3
        echo PROCESSING_CONCURRENT_FILES=5
        echo PROCESSING_MIN_IMAGE_SIZE=50
        echo.
        echo # Directory Configuration
        echo TEMP_DIR=./temp
        echo OUTPUT_DIR=./output
        echo LOG_DIR=./logs
        echo CONFIG_SCHEMA_PATH=./schema/config_schema.json
    ) > .env
    echo [93mPlease edit .env file with your OpenAI API key and other settings[0m
    start notepad .env
)

:: Check if running with input directory
set "INPUT_DIR=%~1"
if "%INPUT_DIR%"=="" (
    echo.
    echo [93mUsage: run.bat [input_directory] [options][0m
    echo.
    echo Options:
    echo   --output, -o    Output file path ^(default: output/config.json^)
    echo   --schema, -s    Custom schema file path
    echo   --concurrent, -c Number of concurrent files to process
    echo.
    set /p "INPUT_DIR=Please enter input directory path: "
)

:: Remove quotes if present
set INPUT_DIR=!INPUT_DIR:"=!

:: Validate input directory
if not exist "!INPUT_DIR!" (
    echo [91mError: Input directory does not exist: !INPUT_DIR![0m
    pause
    exit /b 1
)

:: Set default output file if not specified
set "OUTPUT_FILE=output\config.json"
set "SCHEMA_FILE=schema\config_schema.json"
set "CONCURRENT_FILES=5"

:: Parse command line arguments
:parse_args
if "%~2"=="" goto :end_parse_args
if /i "%~2"=="-o" set "OUTPUT_FILE=%~3"
if /i "%~2"=="--output" set "OUTPUT_FILE=%~3"
if /i "%~2"=="-s" set "SCHEMA_FILE=%~3"
if /i "%~2"=="--schema" set "SCHEMA_FILE=%~3"
if /i "%~2"=="-c" set "CONCURRENT_FILES=%~3"
if /i "%~2"=="--concurrent" set "CONCURRENT_FILES=%~3"
shift
goto :parse_args
:end_parse_args

call :log "Starting configuration extraction..."
echo [92mStarting configuration extraction...[0m
echo Input Directory: !INPUT_DIR!
echo Output File: !OUTPUT_FILE!
echo Schema File: !SCHEMA_FILE!
echo Concurrent Files: !CONCURRENT_FILES!
echo.

:: Ensure all paths use forward slashes for consistency
set "INPUT_DIR=!INPUT_DIR:\=/!"
set "OUTPUT_FILE=!OUTPUT_FILE:\=/!"
set "SCHEMA_FILE=!SCHEMA_FILE:\=/!"

:: Run with all required arguments
python config_extractor.py "!INPUT_DIR!" --output "!OUTPUT_FILE!" --schema "!SCHEMA_FILE!" --concurrent-files !CONCURRENT_FILES!
if !ERRORLEVEL! NEQ 0 (
    echo [91mError occurred during extraction[0m
    call :log "Error occurred during extraction"
    pause
    exit /b 1
)

:: Log function
:log
echo %date% %time% - %~1 >> "%INSTALL_LOG%"
goto :eof