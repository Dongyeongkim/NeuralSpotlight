; DigUp's additions to Tauri's NSIS installer (bundle.windows.nsis.installerHooks).

; "Delete the application data" also takes DigUp's downloaded model files and its index: the app keeps them in
; %LOCALAPPDATA%\DigUp, not in the data folders named after its identifier, which Tauri deletes. Never on an update.
!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $DeleteAppDataCheckboxState = 1
  ${AndIf} $UpdateMode <> 1
    RMDir /r "$LOCALAPPDATA\DigUp\Models"
    RMDir /r "$LOCALAPPDATA\DigUp\index"
    RMDir "$LOCALAPPDATA\DigUp"
  ${EndIf}
!macroend
