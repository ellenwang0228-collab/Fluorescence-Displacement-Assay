#!/usr/bin/env python3
"""
PhosphoMAX launcher — the thing behind the icon.

Run by PhosphoMAX.app when you double-click it. Its whole job is to find a
Python that can import PySide6, then hand over to fda_launcher.py.

Why it is not simply `python3 fda_launcher.py`:

  * A double-clicked .app inherits none of your shell setup. No ~/.zshrc, no
    conda activate, no PATH additions. `python3` may not even be the Python
    that has PySide6 installed — on a stock Mac it is Apple's system Python,
    which does not.

  * A double-clicked .app has no terminal. Anything printed to stderr goes
    nowhere, so a crash looks like the icon bouncing once and giving up. Every
    failure here is therefore reported in a real dialog box and written to a
    log file.

This file is plain Python 3 with no third-party imports, so it runs under
whatever interpreter macOS gives it while it looks for a better one.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "PhosphoMAX"

# Written next to the app so the (slow) search happens once, not every launch.
CONFIG_PATH = os.path.join(HERE, ".phosphomax_launcher.json")
LOG_PATH = os.path.join(HERE, "launcher.log")

# Set by the .app stub; falls back to sitting beside this file.
TARGET = os.environ.get("PHOSPHOMAX_TARGET") or os.path.join(
    HERE, "fda_launcher.py")


# ── logging ───────────────────────────────────────────────────────────────────

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, file=sys.stderr)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# ── user-visible errors ───────────────────────────────────────────────────────

def alert(title, message, detail=""):
    """Show a real dialog. osascript is present on every Mac, and unlike a
    print() it is actually seen when the app was launched from Finder."""
    body = message if not detail else f"{message}\n\n{detail}"
    body = body.replace("\\", "\\\\").replace('"', '\\"')
    title_esc = title.replace('"', '\\"')
    script = (f'display dialog "{body}" with title "{title_esc}" '
              f'buttons {{"Show Log", "OK"}} default button "OK" '
              f'with icon caution')
    try:
        r = subprocess.run(["osascript", "-e", script],
                           capture_output=True, text=True, timeout=300)
        if "Show Log" in (r.stdout or ""):
            subprocess.run(["open", "-R", LOG_PATH], check=False)
    except Exception:
        log(f"could not show dialog: {title}: {message}")


# ── finding a usable interpreter ──────────────────────────────────────────────

PROBE = (
    "import sys, PySide6, matplotlib, pandas, numpy, scipy; "
    "sys.stdout.write(sys.executable)"
)


def works(python_exe, timeout=60):
    """True if this interpreter can import everything the app needs."""
    if not python_exe or not os.path.isfile(python_exe):
        return False
    if not os.access(python_exe, os.X_OK):
        return False
    try:
        r = subprocess.run([python_exe, "-c", PROBE],
                           capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        log(f"  probe error {python_exe}: {type(exc).__name__}")
        return False
    if r.returncode == 0:
        return True
    first = (r.stderr or "").strip().splitlines()
    log(f"  no: {python_exe} — {first[-1] if first else 'unknown error'}")
    return False


def candidates():
    """Every plausible interpreter, best guesses first.

    Ordering matters: an env the user actually activated beats a generic one,
    and a conda env beats Apple's system Python, which never has PySide6.
    """
    home = os.path.expanduser("~")
    seen, out = set(), []

    def add(p):
        if p and p not in seen:
            seen.add(p)
            out.append(p)

    # 1. anything the environment already points at
    add(os.environ.get("PHOSPHOMAX_PYTHON"))
    if os.environ.get("CONDA_PREFIX"):
        add(os.path.join(os.environ["CONDA_PREFIX"], "bin", "python3"))
    if os.environ.get("VIRTUAL_ENV"):
        add(os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python3"))

    # 2. every environment of every conda install we can find
    for base in (f"{home}/anaconda3", f"{home}/miniconda3",
                 f"{home}/miniforge3", f"{home}/mambaforge",
                 f"{home}/opt/anaconda3", f"{home}/opt/miniconda3",
                 "/opt/anaconda3", "/opt/miniconda3", "/opt/homebrew/Caskroom/miniforge/base"):
        envs = os.path.join(base, "envs")
        if os.path.isdir(envs):
            try:
                for env in sorted(os.listdir(envs)):
                    add(os.path.join(envs, env, "bin", "python3"))
            except OSError:
                pass
        add(os.path.join(base, "bin", "python3"))

    # 3. common virtualenv locations inside the project
    for venv in ("venv", ".venv", "env"):
        add(os.path.join(HERE, venv, "bin", "python3"))
        add(os.path.join(os.path.dirname(TARGET), venv, "bin", "python3"))

    # 4. Homebrew and python.org framework builds, newest first
    for root in ("/opt/homebrew/bin", "/usr/local/bin"):
        if os.path.isdir(root):
            try:
                vers = sorted(
                    (f for f in os.listdir(root)
                     if f.startswith("python3.") and f[8:].isdigit()),
                    key=lambda f: int(f[8:]), reverse=True)
                for f in vers:
                    add(os.path.join(root, f))
            except OSError:
                pass
    fw = "/Library/Frameworks/Python.framework/Versions"
    if os.path.isdir(fw):
        try:
            for v in sorted(os.listdir(fw), reverse=True):
                add(os.path.join(fw, v, "bin", "python3"))
        except OSError:
            pass

    # 5. whatever is on PATH, and finally the interpreter running this file
    for name in ("python3", "python"):
        try:
            p = subprocess.run(["/usr/bin/which", name],
                               capture_output=True, text=True, timeout=10)
            if p.returncode == 0:
                add(p.stdout.strip())
        except Exception:
            pass
    add(sys.executable)
    return out


def load_cached():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            return json.load(fh).get("python")
    except Exception:
        return None


def save_cached(python_exe):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump({"python": python_exe,
                       "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "note": "Delete this file to make the launcher "
                               "search for Python again."},
                      fh, indent=2)
    except Exception as exc:
        log(f"could not cache interpreter: {exc}")


def find_python():
    cached = load_cached()
    if cached:
        log(f"trying cached interpreter: {cached}")
        if works(cached):
            return cached, True
        log("cached interpreter no longer works — searching again")

    log("searching for a Python with PySide6…")
    for exe in candidates():
        log(f"  trying {exe}")
        if works(exe):
            log(f"found: {exe}")
            save_cached(exe)
            return exe, False
    return None, False


# ── main ──────────────────────────────────────────────────────────────────────

MISSING_MSG = (
    "PhosphoMAX could not find a Python installation with the packages it "
    "needs.\n\n"
    "It needs an environment with:\n"
    "    PySide6, matplotlib, pandas, numpy, scipy, seaborn, openpyxl\n\n"
    "If you already have one (for example a conda environment you normally "
    "activate before running the script), you can point the app straight at "
    "it by running this once in Terminal:\n\n"
    "    echo '{\"python\": \"/full/path/to/bin/python3\"}' > "
    "\"$CONFIG\"\n\n"
    "To find that path, activate your environment and run:  which python3"
)


def main():
    try:
        open(LOG_PATH, "w").close()      # one launch per log
    except Exception:
        pass
    log(f"{APP_NAME} launcher starting")
    log(f"target: {TARGET}")

    if not os.path.isfile(TARGET):
        alert(f"{APP_NAME} — file missing",
              "The analysis script could not be found.",
              f"Expected it at:\n{TARGET}\n\nIf you moved or renamed the "
              f"FDA_launcher folder, move the app back into it.")
        return 1

    python_exe, was_cached = find_python()
    if not python_exe:
        alert(f"{APP_NAME} — Python not found",
              MISSING_MSG.replace("$CONFIG", CONFIG_PATH),
              f"Details were written to:\n{LOG_PATH}")
        return 1

    log(f"launching with {python_exe}")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Qt occasionally fails to locate its plugins when started outside a
    # shell; pointing at the interpreter's own copy avoids the "could not
    # find the Qt platform plugin cocoa" failure.
    try:
        r = subprocess.run(
            [python_exe, "-c",
             "import PySide6, os; print(os.path.join("
             "os.path.dirname(PySide6.__file__), 'Qt', 'plugins'))"],
            capture_output=True, text=True, timeout=60)
        plugins = r.stdout.strip()
        if r.returncode == 0 and os.path.isdir(plugins):
            env.setdefault("QT_PLUGIN_PATH", plugins)
            log(f"QT_PLUGIN_PATH={plugins}")
    except Exception:
        pass

    try:
        proc = subprocess.run([python_exe, TARGET],
                              cwd=os.path.dirname(TARGET), env=env,
                              capture_output=True, text=True)
    except Exception as exc:
        alert(f"{APP_NAME} — could not start",
              f"{type(exc).__name__}: {exc}", f"Log:\n{LOG_PATH}")
        return 1

    if proc.stdout:
        log("stdout:\n" + proc.stdout)
    if proc.stderr:
        log("stderr:\n" + proc.stderr)

    if proc.returncode != 0:
        stderr = proc.stderr or ""
        tail = stderr.strip().splitlines()
        tail = "\n".join(tail[-12:]) if tail else "(no error output)"
        # A cached interpreter that has since lost a package should be
        # forgotten so the next launch looks elsewhere. But an ordinary bug in
        # the analysis code must NOT invalidate it, or every crash would cost
        # a full re-search. Only environment-shaped failures clear the cache.
        env_failure = any(s in stderr for s in (
            "ModuleNotFoundError", "ImportError", "DLL load failed",
            "Qt platform plugin", "libpython", "Symbol not found"))
        if was_cached and env_failure:
            try:
                os.remove(CONFIG_PATH)
                log("cleared cached interpreter — looks like a broken "
                    "environment, will search again next launch")
            except Exception:
                pass
        alert(f"{APP_NAME} stopped unexpectedly",
              f"The analysis tool exited with code {proc.returncode}.",
              f"{tail}\n\nFull log:\n{LOG_PATH}")
        return proc.returncode

    log("exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
