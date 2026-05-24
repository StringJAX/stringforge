import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_stringforge_is_lightweight_and_warning_clean():
    code = textwrap.dedent(
        """
        import json
        import sys

        import stringforge

        optional_modules = {
            name: name in sys.modules
            for name in ("jaxvacua", "kahlerjax", "jaxiverse", "cytools")
        }
        print(json.dumps({
            "version": stringforge.__version__,
            "optional_modules": optional_modules,
        }))
        """
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "import stringforge should succeed under -W error without optional "
        "solver side effects.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["optional_modules"] == {
        "jaxvacua": False,
        "kahlerjax": False,
        "jaxiverse": False,
        "cytools": False,
    }
    assert "Encountered unexpected exception importing solver" not in result.stderr
    assert "TODO:" not in result.stderr
