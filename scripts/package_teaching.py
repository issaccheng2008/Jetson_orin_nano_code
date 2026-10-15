"""Create a small independent Jetson deployment without copying policy models."""
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "humanoid_jetson_deploy"
OUTPUT = ROOT / "artifacts" / "teaching"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / "jetson-teaching.tar.gz"
    files = [DEPLOY / name for name in (
        "config.py", "protocol.py", "serial_link.py", "motor_test_common.py",
        "teaching.py", "teaching_console.py")]
    files += sorted((DEPLOY / "teaching_web").glob("*"))
    files += [DEPLOY / "tests" / name for name in (
        "test_teaching.py", "test_teaching_serial_pty.py", "test_teaching_http.py")]
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            if path.is_file():
                tar.add(path, arcname=path.relative_to(DEPLOY))
    print(archive)


if __name__ == "__main__":
    main()
