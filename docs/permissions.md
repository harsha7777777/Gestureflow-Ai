# Operating system permissions

GestureFlow AI needs a few permissions to see your webcam and control your computer.
Detailed, screenshot-level steps will be added and verified as each feature lands.

## Windows

- **Camera:** Settings → Privacy & security → Camera → turn on *Camera access* and
  *Let desktop apps access your camera*. (Verified in Milestone 1.)

## macOS

Permissions are granted to the app that *runs Python*, for example Terminal, iTerm or VS Code,
not to Python itself.

- **Camera:** System Settings → Privacy & Security → Camera. (Milestone 1.)
- **Accessibility:** needed to send keyboard and media keys. (Milestone 7.)
- **Screen Recording:** needed for screenshots. (Milestone 7.)

After changing a permission, quit and reopen the terminal or IDE.
