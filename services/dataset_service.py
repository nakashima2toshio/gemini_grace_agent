#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dataset_service.py - データセット操作サービス
=============================================
データセットのダウンロード、読み込み、前処理を担当

機能:
- HuggingFaceデータセットのダウンロード
- Livedoorコーパスのダウンロード・読み込み
- アップロードファイルの読み込み
- テキストの前処理・抽出
"""

import logging
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import pandas as pd

from helper_text import clean_text  # helper_ragではなくhelper_textから直接インポート（循環参照回避）

logger = logging.getLogger(__name__)


def download_livedoor_corpus(save_dir: str = "datasets") -> str:
    """
    Livedoorニュースコーパスをダウンロード

    Args:
        save_dir: 保存ディレクトリ

    Returns:
        解凍後のデータディレクトリパス
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    tar_filename = "ldcc-20140209.tar.gz"
    tar_path = save_path / tar_filename

    # ダウンロード
    if not tar_path.exists():
        logger.info(f"Livedoorニュースコーパスをダウンロード中: {url}")
        urllib.request.urlretrieve(url, tar_path)
        logger.info(f"ダウンロード完了: {tar_path}")

    # 解凍
    extract_path = save_path / "livedoor"
    if not extract_path.exists():
        logger.info("アーカイブを解凍中...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(save_path)
        logger.info("解凍完了")

    # textディレクトリを探す
    text_dir = extract_path / "text"
    if not text_dir.exists():
        text_dir = save_path / "text"

    return str(text_dir)


def load_livedoor_corpus(data_dir: str) -> pd.DataFrame:
    """
    Livedoorコーパスを読み込み

    Args:
        data_dir: データディレクトリパス

    Returns:
        DataFrameとして読み込まれたデータ
    """
    data_path = Path(data_dir)
    records = []

    # カテゴリディレクトリを走査
    for category_dir in data_path.iterdir():
        if not category_dir.is_dir():
            continue

        if category_dir.name in ["CHANGES.txt", "README.txt", "LICENSE.txt"]:
            continue

        category = category_dir.name

        # 記事ファイルを読み込み
        for article_file in category_dir.glob("*.txt"):
            if article_file.name.startswith("LICENSE"):
                continue

            try:
                with open(article_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Livedoor形式: 1行目=URL, 2行目=日付, 3行目=タイトル, 残り=本文
                if len(lines) >= 4:
                    url = lines[0].strip()
                    date = lines[1].strip()
                    title = lines[2].strip()
                    content = "".join(lines[3:]).strip()

                    records.append(
                        {
                            "url": url,
                            "date": date,
                            "title": title,
                            "content": content,
                            "category": category,
                        }
                    )
            except Exception as e:
                logger.warning(f"ファイル読み込みエラー {article_file}: {e}")
                continue

    df = pd.DataFrame(records)
    logger.info(f"Livedoorコーパス読み込み完了: {len(df)} 件")
    return df


def download_hf_dataset(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    sample_size: int,
    log_callback: Callable[[str], None],
) -> pd.DataFrame:
    """
    HuggingFaceからデータセットをダウンロード

    Args:
        dataset_name: データセット名
        config_name: コンフィグ名
        split: データ分割
        sample_size: サンプルサイズ
        log_callback: ログコールバック関数

    Returns:
        DataFrame
    """
    from datasets import load_dataset as hf_load_dataset

    samples = []

    if dataset_name == "wikimedia/wikipedia":
        actual_config = config_name if config_name else "20231101.ja"
        log_callback(f"📥 {dataset_name} をロード中 (config: {actual_config})...")
        dataset = hf_load_dataset(
            dataset_name, actual_config, split=split, streaming=True
        )

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
            samples.append(item)
            if (i + 1) % 100 == 0:
                log_callback(f"進捗: {i + 1}/{sample_size} 件")

    elif dataset_name == "range3/cc100-ja":
        log_callback(f"📥 {dataset_name} をロード中...")
        dataset = hf_load_dataset(dataset_name, split=split, streaming=True)

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
            samples.append(item)
            if (i + 1) % 100 == 0:
                log_callback(f"進捗: {i + 1}/{sample_size} 件")

    elif dataset_name == "cc_news":
        log_callback(f"📥 {dataset_name} をロード中...")
        if config_name:
            dataset = hf_load_dataset(
                dataset_name, config_name, split=split, streaming=True
            )
        else:
            dataset = hf_load_dataset(dataset_name, split=split, streaming=True)

        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
            samples.append(item)
            if (i + 1) % 50 == 0:
                log_callback(f"進捗: {i + 1}/{sample_size} 件")

    else:
        raise ValueError(f"未対応のデータセット: {dataset_name}")

    df = pd.DataFrame(samples)
    log_callback(f"✅ {len(df)} 件のデータをロードしました")
    return df


def extract_text_content(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    データセットからテキストコンテンツを抽出

    Args:
        df: 元のDataFrame
        config: データセット設定（text_field, title_fieldを含む）

    Returns:
        Combined_Textカラムを含むDataFrame
    """
    text_field = config["text_field"]
    title_field = config.get("title_field")

    df_processed = df.copy()

    # タイトルとテキストを結合
    if title_field and title_field in df.columns and text_field in df.columns:
        df_processed["Combined_Text"] = df_processed.apply(
            lambda row: f"{clean_text(str(row.get(title_field, '')))} {clean_text(str(row.get(text_field, '')))}".strip(),
            axis=1,
        )
    elif text_field in df.columns:
        df_processed["Combined_Text"] = df_processed[text_field].apply(
            lambda x: clean_text(str(x)) if x is not None else ""
        )
    else:
        # フォールバック: 利用可能なテキストフィールドを探す
        text_candidates = ["text", "content", "body", "document", "abstract"]
        found_field = None
        for field in text_candidates:
            if field in df.columns:
                found_field = field
                break

        if found_field:
            df_processed["Combined_Text"] = df_processed[found_field].apply(
                lambda x: clean_text(str(x)) if x is not None else ""
            )
        else:
            df_processed["Combined_Text"] = df_processed.apply(
                lambda row: " ".join([str(v) for v in row.values if v is not None]),
                axis=1,
            )

    # 空のテキストを除外
    df_processed = df_processed[df_processed["Combined_Text"].str.strip() != ""]

    return df_processed


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    アップロードされたファイルを読み込み

    Args:
        uploaded_file: Streamlitのfile_uploaderで取得したファイル

    Returns:
        Combined_Textカラムを含むDataFrame
    """
    file_extension = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_extension == "csv":
            # CSVファイル
            df = pd.read_csv(uploaded_file)

        elif file_extension in ["txt", "text"]:
            # テキストファイル（1行1ドキュメント）
            content = uploaded_file.read().decode("utf-8")
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            df = pd.DataFrame({"text": lines})

        elif file_extension == "json":
            # JSONファイル
            content = uploaded_file.read().decode("utf-8")
            data = json.loads(content)

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError(
                    "JSONファイルはリストまたはオブジェクトである必要があります"
                )

        elif file_extension == "jsonl":
            # JSON Linesファイル
            content = uploaded_file.read().decode("utf-8")
            lines = [json.loads(line) for line in content.split("\n") if line.strip()]
            df = pd.DataFrame(lines)

        else:
            raise ValueError(f"未対応のファイル形式: {file_extension}")

        # Combined_Textカラムの作成
        if "Combined_Text" not in df.columns:
            # テキストフィールドを探す
            text_candidates = [
                "text",
                "content",
                "body",
                "document",
                "answer",
                "question",
            ]
            found_field = None

            for field in text_candidates:
                if field in df.columns:
                    found_field = field
                    break

            if found_field:
                df["Combined_Text"] = df[found_field].apply(
                    lambda x: clean_text(str(x)) if x is not None else ""
                )
            else:
                # 全カラムを結合
                df["Combined_Text"] = df.apply(
                    lambda row: " ".join([str(v) for v in row.values if v is not None]),
                    axis=1,
                )

        # 空のテキストを除外
        df = df[df["Combined_Text"].str.strip() != ""]
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        raise
