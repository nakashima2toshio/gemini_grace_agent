#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config_service.py - 設定管理サービス
====================================
YAMLベースの設定管理、環境変数オーバーライド、ロギング設定

統合元:
- helper_api.py::ConfigManager
"""

import os
import yaml
import logging
import logging.handlers
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    設定ファイルの管理（シングルトン）

    Features:
    - YAML設定ファイルの読み込み
    - 環境変数によるオーバーライド
    - キャッシュ付き設定値取得
    - ロガー設定
    """

    _instance = None

    def __new__(cls, config_path: str = "config.yml"):
        """シングルトンパターンで設定を管理"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config.yml"):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._cache: Dict[str, Any] = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger('openai_helper')

        # 既に設定済みの場合はスキップ
        if logger.handlers:
            return logger

        log_config = self.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        logger.setLevel(level)

        # フォーマッターの設定
        formatter = logging.Formatter(
            log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ファイルハンドラー（設定されている場合）
        log_file = log_config.get("file")
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_config.get("max_bytes", 10485760),
                backupCount=log_config.get("backup_count", 5)
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    # 環境変数での設定オーバーライド
                    self._apply_env_overrides(config)
                    return config
            except Exception as e:
                print(f"設定ファイルの読み込みに失敗: {e}")
                return self._get_default_config()
        else:
            print(f"設定ファイルが見つかりません: {self.config_path}")
            config = self._get_default_config()
            self._apply_env_overrides(config)
            return config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """環境変数による設定オーバーライド"""
        # OpenAI API Key
        if os.getenv("OPENAI_API_KEY"):
            config.setdefault("api", {})["openai_api_key"] = os.getenv("OPENAI_API_KEY")

        # Google API Key
        if os.getenv("GOOGLE_API_KEY"):
            config.setdefault("api", {})["google_api_key"] = os.getenv("GOOGLE_API_KEY")

        # ログレベル
        if os.getenv("LOG_LEVEL"):
            config.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")

        # デバッグモード
        if os.getenv("DEBUG_MODE"):
            config.setdefault("experimental", {})["debug_mode"] = os.getenv("DEBUG_MODE").lower() == "true"

        # LLMプロバイダー
        if os.getenv("LLM_PROVIDER"):
            config.setdefault("llm", {})["provider"] = os.getenv("LLM_PROVIDER")

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "models": {
                "default": "gemini-2.0-flash",
                "available": ["gemini-2.0-flash", "gemini-2.0-pro", "gpt-4o-mini", "gpt-4o"]
            },
            "api": {
                "timeout": 30,
                "max_retries": 3,
                "openai_api_key": None,
                "google_api_key": None
            },
            "ui": {
                "page_title": "RAG Q/A Generator",
                "page_icon": "🤖",
                "layout": "wide"
            },
            "cache": {
                "enabled": True,
                "ttl": 3600,
                "max_size": 100
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,
                "max_bytes": 10485760,
                "backup_count": 5
            },
            "error_messages": {
                "general_error": "エラーが発生しました",
                "api_key_missing": "APIキーが設定されていません",
                "network_error": "ネットワークエラーが発生しました"
            },
            "experimental": {
                "debug_mode": False,
                "performance_monitoring": True
            },
            "llm": {
                "provider": "gemini"
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値の取得（キャッシュ付き）

        Args:
            key: ドット区切りのキー (例: "api.timeout")
            default: デフォルト値

        Returns:
            設定値
        """
        if key in self._cache:
            return self._cache[key]

        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = default
                break

        result = value if value is not None else default
        self._cache[key] = result
        return result

    def set(self, key: str, value: Any) -> None:
        """
        設定値の更新

        Args:
            key: ドット区切りのキー
            value: 設定する値
        """
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

        # キャッシュクリア
        self._cache.pop(key, None)

    def reload(self) -> None:
        """設定の再読み込み"""
        self._config = self._load_config()
        self._cache.clear()

    def save(self, filepath: str = None) -> bool:
        """
        設定をファイルに保存

        Args:
            filepath: 保存先パス（省略時は元のパス）

        Returns:
            成功時True
        """
        try:
            save_path = Path(filepath) if filepath else self.config_path
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"設定保存エラー: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """全設定を取得"""
        return self._config.copy()

    def has(self, key: str) -> bool:
        """キーが存在するか確認"""
        return self.get(key) is not None


# ===================================================================
# グローバルインスタンス
# ===================================================================

# デフォルト設定インスタンス
config = ConfigManager("config.yml")

# グローバルロガー
logger = config.logger


# ===================================================================
# ユーティリティ関数
# ===================================================================

def get_config(key: str, default: Any = None) -> Any:
    """設定値を取得するショートカット関数"""
    return config.get(key, default)


def set_config(key: str, value: Any) -> None:
    """設定値を更新するショートカット関数"""
    config.set(key, value)


def reload_config() -> None:
    """設定を再読み込みするショートカット関数"""
    config.reload()


# ===================================================================
# エクスポート
# ===================================================================

__all__ = [
    # クラス
    "ConfigManager",
    # グローバルインスタンス
    "config",
    "logger",
    # ユーティリティ関数
    "get_config",
    "set_config",
    "reload_config",
]