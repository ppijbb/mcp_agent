"""Tests for base/environment config merging and config path resolution."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from srcs.core.config import loader


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.original_path = loader._config_path
        self.original_config = loader._config

    def tearDown(self):
        loader._config_path = self.original_path
        loader._config = self.original_config

    def test_environment_config_overrides_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            (config_dir / "base.yaml").write_text(
                "logging:\n  level: INFO\n  log_file: logs/base.log\n"
                "security:\n  allowed_hosts: ['*']\n",
                encoding="utf-8",
            )
            (config_dir / "development.yaml").write_text(
                "logging:\n  level: DEBUG\nsecurity:\n  allowed_hosts: ['localhost']\n",
                encoding="utf-8",
            )

            loader._config_path = config_dir
            loader._config = None
            config = loader.load_config()

            self.assertEqual(config.logging.level, "DEBUG")
            self.assertEqual(config.logging.log_file, "logs/base.log")
            self.assertEqual(config.security.allowed_hosts, ["localhost"])

    def test_default_config_path_is_project_root_relative(self):
        env = dict(os.environ)
        env.pop("MCP_CONFIG_PATH", None)
        env["MCP_ENV"] = "development"
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        result = subprocess.run(
            [sys.executable, "-c", "from srcs.core.config.loader import settings; print(settings.logging.level)"],
            cwd=tempfile.gettempdir(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "DEBUG")
        self.assertNotIn("Configuration files not found", result.stderr)

    def test_explicit_config_path_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "base.yaml").write_text("app_name: custom_app\n", encoding="utf-8")
            env = dict(os.environ)
            env["MCP_CONFIG_PATH"] = tmp
            env["MCP_ENV"] = "development"
            env["PYTHONPATH"] = str(PROJECT_ROOT)

            result = subprocess.run(
                [sys.executable, "-c", "from srcs.core.config.loader import settings; print(settings.app_name)"],
                cwd=tempfile.gettempdir(),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "custom_app")


if __name__ == "__main__":
    unittest.main()
