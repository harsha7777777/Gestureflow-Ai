# Operating system permissions

GestureFlow AI needs a few permissions to see your webcam and control your computer.
Detailed, screenshot-level steps will be added and verified as each feature lands.

## macOS

Permissions are granted to the app that *runs Python*, for example Terminal, iTerm or VS Code,
not to Python itself.

- **Camera:** System Settings → Privacy & Security → Camera. (Milestone 1.)
- **Accessibility:** needed to send keyboard and media keys. (Milestone 4.)
- **Input Monitoring:** may be needed for the global emergency hotkey. (Milestone 4.)
- **Screen Recording:** needed for screenshots. (Milestone 4.)

After changing a permission, quit and reopen the terminal or IDE.

## Windows

- **Camera:** Settings → Privacy & security → Camera → turn on *Camera access* and
  *Let desktop apps access your camera*. (Milestone 7, when Windows support lands.)
