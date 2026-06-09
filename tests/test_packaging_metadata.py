import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_assignment(text, name):
    match = re.search(rf"{name}\s*=\s*['\"]([^'\"]+)['\"]", text)
    assert match is not None, f"Could not find {name!r} assignment."
    return match.group(1)


def test_declared_versions_are_consistent():
    init_text = (REPO_ROOT / "stringforge" / "__init__.py").read_text()
    pyproject_data = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text()
    )
    conf_text = (REPO_ROOT / "documentation" / "source" / "conf.py").read_text()

    versions = {
        "__init__.py": _extract_assignment(init_text, "__version__"),
        "pyproject.toml": pyproject_data["project"]["version"],
        "documentation/source/conf.py": _extract_assignment(conf_text, "version"),
    }

    assert len(set(versions.values())) == 1, versions


def test_pypi_metadata_does_not_depend_on_git_urls():
    setup_text = (REPO_ROOT / "setup.py").read_text()

    assert "@git+" not in setup_text
    assert "git+https://" not in setup_text


def test_core_runtime_dependencies_include_huggingface_hub():
    pyproject_data = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text()
    )

    deps = set(pyproject_data["project"]["dependencies"])
    assert "huggingface_hub" in deps


def test_test_suite_uses_project_license_header():
    test_text = (REPO_ROOT / "tests" / "test_vacua_vault.py").read_text()

    assert "Apache License" not in test_text
    assert "GNU General Public License" in test_text
