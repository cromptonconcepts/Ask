' ============================================================
' TTM Ask - Silent Launcher
' Double-click this file to run setup if needed, start services, and open the app
' ============================================================

Dim oShell, appDir, runScript, command

appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
runScript = appDir & "run_ttm_ask.ps1"

Set oShell = CreateObject("WScript.Shell")
command = "powershell -ExecutionPolicy Bypass -File " & Chr(34) & runScript & Chr(34)
oShell.Run command, 0, False
