# Operating system permissions

GestureFlow AI needs a few permissions to see your webcam and control your computer.
Detailed, screenshot-level steps will be added and verified as each feature lands.

## macOS

Permissions are granted to the app that *runs Python*, for example Terminal, iTerm or VS Code,
not to Python itself.

- **Camera** (needed from Milestone 1):
  1. Run `python -m gestureflow` from Terminal. macOS asks *"Terminal" would like to access the
     camera*: click **Allow**.
  2. Quit Terminal completely (Cmd+Q; closing the window is not enough), reopen it, activate
     the environment and run the app again. macOS only applies the permission after a restart
     of the app.
  3. If you clicked *Don't Allow* or never saw the prompt: open System Settings → Privacy &
     Security → Camera, turn on Terminal (or iTerm / VS Code), then do step 2.
  4. To reset and get the prompt again: `tccutil reset Camera com.apple.Terminal`
- **Accessibility:** needed to send keyboard and media keys. (Milestone 4.)
- **Input Monitoring:** may be needed for the global emergency hotkey. (Milestone 4.)
- **Screen Recording:** needed for screenshots. (Milestone 4.)

After changing a permission, quit and reopen the terminal or IDE.

## Windows

- **Camera:** Settings → Privacy & security → Camera → turn on *Camera access* and
  *Let desktop apps access your camera*. (Milestone 7, when Windows support lands.)
