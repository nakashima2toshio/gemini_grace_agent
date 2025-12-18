# python sample_regex_mecab.py
# MeCab複合名詞版と正規表現版を統合したロバストなキーワード抽出システム

import re
from typing import List, Dict, Tuple
from collections import Counter


class KeywordExtractor:
    """
    MeCabと正規表現を統合したキーワード抽出クラス

    MeCabが利用可能な場合は複合名詞抽出を優先し、
    利用不可の場合は正規表現版に自動フォールバック
    """

    def __init__(self, prefer_mecab: bool = True):
        """
        Args:
            prefer_mecab: MeCabを優先的に使用するか（デフォルト: True）
        """
        self.prefer_mecab = prefer_mecab
        self.mecab_available = self._check_mecab_availability()

        # ストップワード定義
        self.stopwords = {
            'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
            'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
            'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と',
            'の', 'から', 'まで', '等', 'など', 'よる', 'おく', 'くる'
        }

        # 重要キーワードの定義（スコアブースト用）
        self.important_keywords = {
            'AI', '人工知能', '機械学習', '深層学習', 'ディープラーニング',
            '自然言語処理', 'NLP', 'トランスフォーマー', 'BERT', 'GPT',
            'CNN', 'Vision', 'Transformer', '医療', '診断', '自動運転',
            '倫理', 'バイアス', '課題', '問題', 'モデル', 'データ'
        }

        if self.mecab_available:
            print("✅ MeCabが利用可能です（複合名詞抽出モード）")
        else:
            print("⚠️ MeCabが利用できません（正規表現モード）")

    def _check_mecab_availability(self) -> bool:
        """MeCabの利用可能性をチェック"""
        try:
            import MeCab
            # 実際にインスタンス化して動作確認
            tagger = MeCab.Tagger()
            tagger.parse("テスト")
            return True
        except (ImportError, RuntimeError):
            return False

    def extract(self, text: str, top_n: int = 5,
                use_scoring: bool = True) -> List[str]:
        """
        テキストからキーワードを抽出（自動フォールバック対応）

        Args:
            text: 分析対象テキスト
            top_n: 抽出するキーワード数
            use_scoring: スコアリングを使用するか

        Returns:
            キーワードリスト
        """
        if self.mecab_available and self.prefer_mecab:
            try:
                keywords = self._extract_with_mecab(text, top_n, use_scoring)
                if keywords:  # 空でなければ成功
                    return keywords
            except Exception as e:
                print(f"⚠️ MeCab抽出エラー: {e}")

        # フォールバック: 正規表現版
        return self._extract_with_regex(text, top_n, use_scoring)

    def _extract_with_mecab(self, text: str, top_n: int,
                           use_scoring: bool) -> List[str]:
        """MeCabを使用した複合名詞抽出"""
        import MeCab

        tagger = MeCab.Tagger()
        node = tagger.parseToNode(text)

        # 複合名詞の抽出
        compound_buffer = []
        compound_nouns = []

        while node:
            features = node.feature.split(',')
            pos = features[0]  # 品詞

            if pos == '名詞':
                compound_buffer.append(node.surface)
            else:
                # 名詞以外が来たらバッファをフラッシュ
                if compound_buffer:
                    compound_noun = ''.join(compound_buffer)
                    if len(compound_noun) > 0:
                        compound_nouns.append(compound_noun)
                    compound_buffer = []

            node = node.next

        # 最後のバッファをフラッシュ
        if compound_buffer:
            compound_noun = ''.join(compound_buffer)
            if len(compound_noun) > 0:
                compound_nouns.append(compound_noun)

        # フィルタリングとスコアリング
        if use_scoring:
            return self._score_and_rank(compound_nouns, top_n)
        else:
            return self._filter_and_count(compound_nouns, top_n)

    def _extract_with_regex(self, text: str, top_n: int,
                           use_scoring: bool) -> List[str]:
        """正規表現を使用したキーワード抽出"""
        # カタカナ語、漢字複合語、英数字を抽出
        pattern = r'[ァ-ヴー]{2,}|[一-龥]{2,}|[A-Za-z]{2,}[A-Za-z0-9]*'
        words = re.findall(pattern, text)

        # フィルタリングとスコアリング
        if use_scoring:
            return self._score_and_rank(words, top_n)
        else:
            return self._filter_and_count(words, top_n)

    def _filter_and_count(self, words: List[str], top_n: int) -> List[str]:
        """頻度ベースのフィルタリング（シンプル版）"""
        # ストップワード除外
        filtered = [w for w in words if w not in self.stopwords and len(w) > 1]

        # 頻度カウント
        word_freq = Counter(filtered)

        # 上位N件を返す
        return [word for word, freq in word_freq.most_common(top_n)]

    def _score_and_rank(self, words: List[str], top_n: int) -> List[str]:
        """スコアリングベースのランキング（高度版）"""
        word_scores = {}
        word_freq = Counter(words)

        for word, freq in word_freq.items():
            # ストップワード除外
            if word in self.stopwords or len(word) <= 1:
                continue

            score = 0.0

            # 1. 頻度スコア（正規化: 最大3回まで）
            freq_score = min(freq / 3.0, 1.0) * 0.3
            score += freq_score

            # 2. 長さスコア（複合語優遇）
            length_score = min(len(word) / 8.0, 1.0) * 0.3
            score += length_score

            # 3. 重要キーワードブースト
            if word in self.important_keywords:
                score += 0.5

            # 4. 文字種スコア
            # カタカナ3文字以上
            if re.match(r'^[ァ-ヴー]{3,}$', word):
                score += 0.2
            # 英大文字2文字以上
            elif re.match(r'^[A-Z]{2,}$', word):
                score += 0.3
            # 漢字4文字以上
            elif re.match(r'^[一-龥]{4,}$', word):
                score += 0.2

            word_scores[word] = score

        # スコア降順でソート
        ranked = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

        return [word for word, score in ranked[:top_n]]

    def extract_with_details(self, text: str, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        詳細情報付きでキーワードを抽出（比較分析用）

        Returns:
            各手法での抽出結果と詳細スコア
        """
        results = {}

        # MeCab複合名詞版
        if self.mecab_available:
            try:
                mecab_keywords = self._extract_with_mecab_scored(text, top_n)
                results['MeCab複合名詞'] = mecab_keywords
            except Exception as e:
                results['MeCab複合名詞'] = [(f"エラー: {e}", 0.0)]

        # 正規表現版
        regex_keywords = self._extract_with_regex_scored(text, top_n)
        results['正規表現'] = regex_keywords

        # 統合版（デフォルト動作）
        integrated_keywords = self._extract_integrated(text, top_n)
        results['統合版'] = integrated_keywords

        return results

    def _extract_with_mecab_scored(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """MeCab版（スコア付き）"""
        keywords = self._extract_with_mecab(text, top_n, use_scoring=True)
        # スコアを再計算
        scored = []
        for kw in keywords:
            score = self._calculate_keyword_score(kw, text)
            scored.append((kw, score))
        return scored

    def _extract_with_regex_scored(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """正規表現版（スコア付き）"""
        keywords = self._extract_with_regex(text, top_n, use_scoring=True)
        scored = []
        for kw in keywords:
            score = self._calculate_keyword_score(kw, text)
            scored.append((kw, score))
        return scored

    def _extract_integrated(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """統合版: MeCabと正規表現の結果をマージ"""
        all_keywords = set()

        # MeCabから抽出
        if self.mecab_available:
            try:
                mecab_kws = self._extract_with_mecab(text, top_n * 2, use_scoring=False)
                all_keywords.update(mecab_kws)
            except Exception:
                pass

        # 正規表現から抽出
        regex_kws = self._extract_with_regex(text, top_n * 2, use_scoring=False)
        all_keywords.update(regex_kws)

        # 統合スコアリング
        scored = []
        for kw in all_keywords:
            if kw in self.stopwords or len(kw) <= 1:
                continue
            score = self._calculate_keyword_score(kw, text)
            scored.append((kw, score))

        # スコア降順でソート
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _calculate_keyword_score(self, keyword: str, text: str) -> float:
        """キーワードの総合スコアを計算"""
        score = 0.0

        # 出現頻度
        freq = text.count(keyword)
        freq_score = min(freq / 3.0, 1.0) * 0.3
        score += freq_score

        # 長さ
        length_score = min(len(keyword) / 8.0, 1.0) * 0.2
        score += length_score

        # 重要キーワード
        if keyword in self.important_keywords:
            score += 0.4

        # 文字種
        if re.match(r'^[ァ-ヴー]{3,}$', keyword):
            score += 0.15
        elif re.match(r'^[A-Z]{2,}$', keyword):
            score += 0.2
        elif re.match(r'^[一-龥]{4,}$', keyword):
            score += 0.15

        return min(score, 1.0)


def compare_methods(text: str, top_n: int = 10):
    """各抽出手法を比較して結果を表示"""
    extractor = KeywordExtractor()

    print("=" * 80)
    print("キーワード抽出手法の比較")
    print("=" * 80)

    results = extractor.extract_with_details(text, top_n)

    for method, keywords in results.items():
        print(f"\n【{method}】")
        print("-" * 80)
        for i, (keyword, score) in enumerate(keywords, 1):
            print(f"  {i:2d}. {keyword:20s} (スコア: {score:.3f})")

    # 共通キーワードの分析
    print("\n" + "=" * 80)
    print("手法間の共通性分析")
    print("=" * 80)

    keyword_sets = {method: set(kw for kw, _ in kws)
                   for method, kws in results.items()}

    # 全手法で共通
    common_all = set.intersection(*keyword_sets.values())
    if common_all:
        print(f"\n全手法で共通: {', '.join(common_all)}")

    # MeCabと統合版で共通
    if 'MeCab複合名詞' in keyword_sets and '統合版' in keyword_sets:
        common_mecab_int = keyword_sets['MeCab複合名詞'] & keyword_sets['統合版']
        if common_mecab_int:
            print(f"MeCab・統合版で共通: {', '.join(common_mecab_int)}")

    # 正規表現と統合版で共通
    if '正規表現' in keyword_sets and '統合版' in keyword_sets:
        common_regex_int = keyword_sets['正規表現'] & keyword_sets['統合版']
        if common_regex_int:
            print(f"正規表現・統合版で共通: {', '.join(common_regex_int)}")


def evaluate_coverage_potential(keywords: List[str],
                                uncovered_text: str,
                                analyzer=None) -> Dict[str, float]:
    """
    キーワードのカバレージ改善ポテンシャルを評価

    Args:
        keywords: 抽出されたキーワード
        uncovered_text: 未カバーのテキスト
        analyzer: SemanticCoverageインスタンス（オプション）

    Returns:
        評価指標の辞書
    """
    metrics = {}

    # 1. キーワードカバレージ率
    covered_count = sum(1 for kw in keywords if kw in uncovered_text)
    metrics['キーワードカバレージ率'] = covered_count / len(keywords) if keywords else 0

    # 2. 複合語率
    compound_count = sum(1 for kw in keywords if len(kw) >= 4)
    metrics['複合語率'] = compound_count / len(keywords) if keywords else 0

    # 3. 専門用語率
    technical_pattern = r'^([ァ-ヴー]{3,}|[A-Z]{2,}|[一-龥]{4,})$'
    technical_count = sum(1 for kw in keywords if re.match(technical_pattern, kw))
    metrics['専門用語率'] = technical_count / len(keywords) if keywords else 0

    # 4. 意味的関連度（analyzerが提供された場合）
    if analyzer is not None:
        try:
            keyword_text = ' '.join(keywords)
            kw_emb = analyzer.generate_embedding(keyword_text)
            text_emb = analyzer.generate_embedding(uncovered_text)
            similarity = analyzer.cosine_similarity(kw_emb, text_emb)
            metrics['意味的関連度'] = similarity
        except Exception:
            metrics['意味的関連度'] = 0.0

    return metrics


def main():
    """メイン実行関数"""

    # サンプルテキスト
    sample_text = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
    画像認識の分野では、CNNが主流でしたが、最近ではVision Transformerも注目されています。
    AIの応用は医療診断から自動運転まで幅広く、社会に大きな影響を与えています。
    しかし、AIの倫理的な課題やバイアスの問題も重要な議論となっています。
    """

    # 基本的な抽出テスト
    print("=" * 80)
    print("基本的なキーワード抽出テスト")
    print("=" * 80)

    extractor = KeywordExtractor()

    # デフォルト抽出
    keywords = extractor.extract(sample_text, top_n=10)
    print("\n【統合版抽出結果（上位10件）】")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i:2d}. {kw}")

    print("\n")

    # 詳細比較
    compare_methods(sample_text, top_n=10)

    # カバレージポテンシャル評価（analyzerなし版）
    print("\n" + "=" * 80)
    print("キーワード品質評価")
    print("=" * 80)

    results = extractor.extract_with_details(sample_text, top_n=10)
    for method, keywords_scored in results.items():
        keywords = [kw for kw, _ in keywords_scored]
        metrics = evaluate_coverage_potential(keywords, sample_text)

        print(f"\n【{method}】")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2%}")


if __name__ == "__main__":
    main()