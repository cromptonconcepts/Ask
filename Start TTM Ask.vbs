' ============================================================
' TTM Ask - Adaptive Launcher
' First run shows setup progress. Later launches stay quiet.
' ============================================================

Dim oShell, fileSystem, appDir, runScript, venvPython, setupState, command, windowStyle

appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
runScript = appDir & "run_ttm_ask.ps1"
venvPython = appDir & ".venv\Scripts\python.exe"
setupState = appDir & "logs\setup_state.json"

Set oShell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

command = "powershell -NoProfile -ExecutionPolicy Bypass -File " & Chr(34) & runScript & Chr(34)
windowStyle = 0

If (Not fileSystem.FileExists(venvPython)) Or (Not fileSystem.FileExists(setupState)) Then
	windowStyle = 1
End If

oShell.Run command, windowStyle, False