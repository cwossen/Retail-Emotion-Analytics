"""Add HSEmotion as a --classifier option in run_video.py and speech_check.py.

Both scripts currently branch on DAN / POSTER++ only. This inserts an
HSEmotion branch and extends the argparse choices, so the same clip can be run
through all three models without touching anything else.

Each file is patched all-or-nothing: if either the loader branch or the
argparse choices line isn't found in the expected form, that file is left
completely untouched - no partial patch that would let --classifier HSEmotion
silently fall through to POSTER++.

Makes a .bak of each file before writing. Idempotent: running it twice is
harmless.

    python3 add_hsemotion.py
"""
import importlib.util
import os
import shutil

ROOT = os.path.expanduser("~/datasets/emotion_sr")
TARGETS = ["run_video.py", "speech_check.py"]
WRAPPER = "hsemotion_wrapper.py"

OLD_BRANCH = '''    if args.classifier == "DAN":
        print("Loading DAN...")
        classify = load_dan()
    else:
        print("Loading POSTER++...")'''

NEW_BRANCH = '''    if args.classifier == "DAN":
        print("Loading DAN...")
        classify = load_dan()
    elif args.classifier == "HSEmotion":
        from hsemotion_wrapper import load_hsemotion
        classify = load_hsemotion()
    else:
        print("Loading POSTER++...")'''

OLD_CHOICES = 'choices=["DAN", "POSTER++"]'
NEW_CHOICES = 'choices=["DAN", "POSTER++", "HSEmotion"]'


def patch(path):
    name = os.path.basename(path)
    if not os.path.isfile(path):
        print(f"  SKIP {name}: not found")
        return
    with open(path) as f:
        src = f.read()

    if "load_hsemotion" in src:
        print(f"  SKIP {name}: already patched")
        return

    # All-or-nothing: both edits must apply, or we touch nothing.
    missing = []
    if OLD_BRANCH not in src:
        missing.append("loader branch")
    if OLD_CHOICES not in src:
        missing.append("argparse choices")
    if missing:
        print(f"  SKIP {name}: {' and '.join(missing)} not found in the "
              f"expected form - file left untouched. Check it by hand.")
        return

    src = src.replace(OLD_BRANCH, NEW_BRANCH).replace(OLD_CHOICES, NEW_CHOICES)
    shutil.copy2(path, path + ".bak")
    with open(path, "w") as f:
        f.write(src)
    print(f"  PATCHED {name}: loader branch + argparse choices  "
          f"(backup at {name}.bak)")


def check_prereqs():
    print("Checking prerequisites:")
    wpath = os.path.join(ROOT, WRAPPER)
    if os.path.isfile(wpath):
        print(f"  OK  {WRAPPER} present")
    else:
        print(f"  MISSING {WRAPPER}: the HSEmotion branch imports "
              "load_hsemotion from it.")
        print(f"          Create {wpath} with a load_hsemotion() that returns")
        print("          the SAME object type load_dan() does - same call")
        print("          signature and label/output shape - or the branch")
        print("          will fail (or mislabel) at runtime.")
    if importlib.util.find_spec("hsemotion") is None:
        print("  MISSING hsemotion package: pip install hsemotion "
              "(or the onnx build).")
    else:
        print("  OK  hsemotion package importable")


def main():
    print("Adding HSEmotion support...")
    for t in TARGETS:
        patch(os.path.join(ROOT, t))

    print("\nVerifying patched files still compile:")
    import py_compile
    ok = True
    for t in TARGETS:
        p = os.path.join(ROOT, t)
        if not os.path.isfile(p):
            continue
        try:
            py_compile.compile(p, doraise=True)
            print(f"  OK  {t}")
        except py_compile.PyCompileError as e:
            ok = False
            print(f"  FAIL {t}: {e}")

    print()
    check_prereqs()

    print()
    if not ok:
        print("A file failed to compile - fix that before going further.")
    else:
        print("Syntax is clean. That is all py_compile proves - not that it "
              "runs.")
        print(f"Make sure {WRAPPER} and the hsemotion package are in place "
              "(see above),")
        print("then push ONE clip through all three classifiers and confirm "
              "HSEmotion")
        print("returns sane, correctly-labeled output before you trust it.")


if __name__ == "__main__":
    main()