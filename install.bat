@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
 echo Python Launcher bulunamadi. https://www.python.org/downloads/ adresinden Python kur.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -m venv .venv
if errorlevel 1 (
 echo Python sanal ortami olusturulamadi.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
 echo Bagimlilik kurulumu basarisiz. Internet baglantisini kontrol et.
 pause
 exit /b 1
)
echo Kurulum tamamlandi. run_auto.bat dosyasini calistirabilirsin.
pause
