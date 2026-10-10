"""Platform action layer: executes a command name on the operating system.

Receives command names only and never touches the ML model. One adapter per OS
(windows.py, macos.py) implements the interface defined in base.py.
"""
