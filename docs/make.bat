@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
        set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=_src/
set BUILDDIR=build
set SPHINXFLAGS=-a -n -A local=1
set SPHINXOPTS=%SPHINXFLAGS% %SOURCE%
set ALLSPHINXOPTS=-d %BUILDDIR%/doctrees %SPHINXOPTS%

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
        echo.
        echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
        echo.installed, then set the SPHINXBUILD environment variable to point
        echo.to the full path of the 'sphinx-build' executable. Alternatively you
        echo.may add the Sphinx directory to PATH.
        echo.
        echo.If you don't have Sphinx installed, grab it from
        echo.http://sphinx-doc.org/
        exit /b 1
)

%SPHINXBUILD% -b %1 %ALLSPINXOPTS% %BUILDDIR%/%1
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd

