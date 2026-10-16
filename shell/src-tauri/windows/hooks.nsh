; NeuralSpotlight's additions to Tauri's NSIS installer (bundle.windows.nsis.installerHooks).

; "Delete the application data" also takes NeuralSpotlight's downloaded model files and its index: the app keeps them in
; %LOCALAPPDATA%\NeuralSpotlight, not in the data folders named after its identifier, which Tauri deletes. Never on an update.
!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $DeleteAppDataCheckboxState = 1
  ${AndIf} $UpdateMode <> 1
    RMDir /r "$LOCALAPPDATA\NeuralSpotlight\Models"
    RMDir /r "$LOCALAPPDATA\NeuralSpotlight\index"
    RMDir "$LOCALAPPDATA\NeuralSpotlight"
  ${EndIf}
!macroend
