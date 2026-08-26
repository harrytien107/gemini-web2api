"""Native Windows system tray host for the HTTP server."""
import ctypes
import os
import subprocess
import sys
import threading
import webbrowser
from ctypes import wintypes

from .gemini import auth_status


WM_DESTROY = 0x0002
WM_COMMAND = 0x0111
WM_USER = 0x0400
WM_RBUTTONUP = 0x0205
WM_LBUTTONDBLCLK = 0x0203
NIM_ADD = 0x00000000
NIM_DELETE = 0x00000002
NIF_MESSAGE = 0x00000001
NIF_ICON = 0x00000002
NIF_TIP = 0x00000004
IDI_APPLICATION = 32512
MF_STRING = 0x00000000
MF_GRAYED = 0x00000001
MF_SEPARATOR = 0x00000800
TPM_RIGHTBUTTON = 0x0002
TPM_RETURNCMD = 0x0100
OPEN_API = 1001
OPEN_CONFIG = 1002
COPY_ENDPOINT = 1003
RESTART = 1005
EXIT = 1006
TRAY_MESSAGE = WM_USER + 1
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128),
        ("dwState", wintypes.DWORD),
        ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * 256),
        ("uVersion", wintypes.UINT),
        ("szInfoTitle", wintypes.WCHAR * 64),
        ("dwInfoFlags", wintypes.DWORD),
        ("guidItem", ctypes.c_byte * 16),
        ("hBalloonIcon", wintypes.HICON),
    ]


WNDPROC = ctypes.WINFUNCTYPE(
    wintypes.LPARAM, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


def _restart_command() -> list:
    args = [arg for arg in sys.argv[1:] if arg != "--tray"]
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "gemini_web2api", "--tray", *args]


def _restart_app() -> None:
    command = _restart_command()
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(command, **kwargs)


def run_tray(server, config_path: str):
    """Run the server in a worker thread and own the Windows tray message loop."""
    user32 = ctypes.windll.user32
    shell32 = ctypes.windll.shell32
    kernel32 = ctypes.windll.kernel32
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    user32.CreatePopupMenu.restype = wintypes.HMENU
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
    ]
    user32.DefWindowProcW.restype = wintypes.LPARAM
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.DestroyMenu.argtypes = [wintypes.HMENU]
    user32.LoadIconW.restype = wintypes.HICON
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
    base_url = f"http://localhost:{server.server_address[1]}/v1"
    api_url = f"{base_url}/models"
    class_name = "GeminiWeb2ApiTray"
    restart_requested = False

    def open_config():
        if config_path and os.path.isfile(config_path):
            os.startfile(os.path.abspath(config_path))
        elif config_path:
            os.startfile(os.path.dirname(os.path.abspath(config_path)) or ".")

    def copy_endpoint(hwnd):
        encoded = (base_url + "\0").encode("utf-16-le")
        if not user32.OpenClipboard(hwnd):
            user32.MessageBoxW(hwnd, "Unable to open the clipboard.", "Copy endpoint", 0x10)
            return
        memory = None
        try:
            user32.EmptyClipboard()
            memory = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
            pointer = kernel32.GlobalLock(memory) if memory else None
            if not pointer:
                raise ctypes.WinError()
            ctypes.memmove(pointer, encoded, len(encoded))
            kernel32.GlobalUnlock(memory)
            if not user32.SetClipboardData(CF_UNICODETEXT, memory):
                raise ctypes.WinError()
            memory = None  # Windows owns the allocation after SetClipboardData succeeds.
        except OSError as error:
            user32.MessageBoxW(hwnd, f"Unable to copy endpoint: {error}", "Copy endpoint", 0x10)
        finally:
            if memory:
                kernel32.GlobalFree(memory)
            user32.CloseClipboard()

    def show_menu(hwnd):
        nonlocal restart_requested
        status = auth_status()
        if status["loaded"] and status["error"]:
            auth_label = "Auth: Cookie loaded (last valid)"
        elif status["loaded"]:
            auth_label = "Auth: Cookie loaded"
        elif status["error"]:
            auth_label = "Auth: Anonymous (auth file error)"
        else:
            auth_label = "Auth: Anonymous"
        menu = user32.CreatePopupMenu()
        try:
            user32.AppendMenuW(menu, MF_STRING | MF_GRAYED, 0, auth_label)
            user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
            user32.AppendMenuW(menu, MF_STRING, OPEN_API, "Open API")
            user32.AppendMenuW(menu, MF_STRING, COPY_ENDPOINT, "Copy endpoint")
            user32.AppendMenuW(menu, MF_STRING, OPEN_CONFIG, "Open config.json")
            user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
            user32.AppendMenuW(menu, MF_STRING, RESTART, "Restart app")
            user32.AppendMenuW(menu, MF_STRING, EXIT, "Exit")
            point = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(point))
            user32.SetForegroundWindow(hwnd)
            command = user32.TrackPopupMenu(
                menu, TPM_RIGHTBUTTON | TPM_RETURNCMD,
                point.x, point.y, 0, hwnd, None,
            )
            if command == OPEN_API:
                webbrowser.open(api_url)
            elif command == COPY_ENDPOINT:
                copy_endpoint(hwnd)
            elif command == OPEN_CONFIG:
                open_config()
            elif command == RESTART:
                restart_requested = True
                user32.PostQuitMessage(0)
            elif command == EXIT:
                user32.PostQuitMessage(0)
        finally:
            user32.DestroyMenu(menu)

    @WNDPROC
    def window_proc(hwnd, message, wparam, lparam):
        if message == TRAY_MESSAGE:
            if lparam == WM_RBUTTONUP:
                show_menu(hwnd)
            elif lparam == WM_LBUTTONDBLCLK:
                webbrowser.open(api_url)
            return 0
        if message == WM_COMMAND:
            return 0
        if message == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    instance = kernel32.GetModuleHandleW(None)
    window_class = WNDCLASSW()
    window_class.lpfnWndProc = window_proc
    window_class.hInstance = instance
    window_class.lpszClassName = class_name
    if not user32.RegisterClassW(ctypes.byref(window_class)):
        raise ctypes.WinError()

    hwnd = user32.CreateWindowExW(
        0, class_name, "gemini-web2api", 0, 0, 0, 0, 0,
        None, None, instance, None,
    )
    if not hwnd:
        raise ctypes.WinError()

    icon_data = NOTIFYICONDATAW()
    icon_data.cbSize = ctypes.sizeof(icon_data)
    icon_data.hWnd = hwnd
    icon_data.uID = 1
    icon_data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
    icon_data.uCallbackMessage = TRAY_MESSAGE
    icon_data.hIcon = user32.LoadIconW(None, IDI_APPLICATION)
    icon_data.szTip = "gemini-web2api"
    if not shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(icon_data)):
        user32.DestroyWindow(hwnd)
        raise ctypes.WinError()

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        message = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
    finally:
        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(icon_data))
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        user32.DestroyWindow(hwnd)
        user32.UnregisterClassW(class_name, instance)
    if restart_requested:
        _restart_app()
