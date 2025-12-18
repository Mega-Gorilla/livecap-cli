"""
文字起こしデータの統一型定義（改訂版）

イベント型ごとに適切な必須フィールドを定義し、
全レイヤーで使用される標準的な辞書フォーマットを提供

Author: Pine Lab
Version: 2.0.0
"""

from typing import TypedDict, Optional, Literal, Any, Dict, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Phase 1 公開面: import * / introspection 用に正式な公開シンボルを定義。
# NOTE: 旧コードが `from livecap_cli.transcription_types import X` を継続利用できるよう、
# 既存の関数名や TypedDict 名は維持したまま公開対象として明示する。
__all__ = [
    # TypedDict 定義
    "TranscriptionEventDict",
    "StatusEventDict",
    "ErrorEventDict",
    "TranslationRequestEventDict",
    "TranslationResultEventDict",
    "SubtitleEventDict",
    "UiEventDict",
    "ExtendedEventDict",
    # 互換レイヤー（将来の削除計画あり）
    "MinimalTranscriptionDict",
    # バリデーション／正規化ユーティリティ
    "validate_event_dict",
    "validate_translation_event",
    "validate_subtitle_event",
    "normalize_to_event_dict",
    "get_event_type_name",
    "format_event_summary",
    # イベント生成ヘルパー
    "create_transcription_event",
    "create_status_event",
    "create_error_event",
    "create_translation_request_event",
    "create_translation_result_event",
    "create_subtitle_event",
]


# === イベント型別の型定義 ===

class TranscriptionEventDict(TypedDict, total=False):
    """
    文字起こしイベントの標準辞書フォーマット
    
    実際のテキスト認識結果を表すイベント。
    """
    
    # === 必須フィールド（transcriptionイベント） ===
    text: str                    # 認識されたテキスト（ソースIDなし）
    source_id: str              # ソース識別子 (source1, source2, etc.)
    is_final: bool              # 確定済みフラグ
    timestamp: float            # Unix timestamp
    event_type: Literal['transcription']  # イベントタイプは固定
    
    # === 基本的な任意フィールド ===
    confidence: Optional[float]  # 信頼度スコア (0.0-1.0)
    language: Optional[str]      # 言語コード (ja, en, zh, etc.)
    phase: Optional[Literal['interim', 'final']]  # 処理フェーズ
    
    # === 表示用フィールド ===
    display_text: Optional[str]  # フォーマット済みテキスト
    
    # === VAD関連フィールド ===
    vad_state: Optional[Literal['silence', 'speech', 'uncertain']]
    speech_probability: Optional[float]  # 0.0-1.0
    
    # === 品質指標 ===
    audio_quality: Optional[float]  # 音声品質スコア
    noise_level: Optional[float]    # ノイズレベル

    # === 拡張用 ===
    metadata: Optional[Dict[str, Any]]  # その他のメタデータ


class StatusEventDict(TypedDict, total=False):
    """
    ステータスイベントの標準辞書フォーマット
    
    処理状態の変更を通知するイベント。
    """
    
    # === 必須フィールド（statusイベント） ===
    event_type: Literal['status']      # イベントタイプは固定
    status_code: str                   # ステータスコード
    message: str                       # ステータスメッセージ
    timestamp: float                   # Unix timestamp
    source_id: str                     # ソース識別子
    
    # === 任意フィールド ===
    phase: Optional[Literal['processing', 'ready', 'idle']]
    metadata: Optional[Dict[str, Any]]


class ErrorEventDict(TypedDict, total=False):
    """
    エラーイベントの標準辞書フォーマット
    
    エラー発生を通知するイベント。
    """
    
    # === 必須フィールド（errorイベント） ===
    event_type: Literal['error']       # イベントタイプは固定
    error_code: str                    # エラーコード
    message: str                       # エラーメッセージ
    timestamp: float                   # Unix timestamp
    source_id: str                     # ソース識別子
    
    # === 任意フィールド ===
    error_details: Optional[str]       # 詳細なエラー情報
    metadata: Optional[Dict[str, Any]]


# === Phase 4: 翻訳・字幕イベント型 ===

class TranslationRequestEventDict(TypedDict, total=False):
    """
    翻訳リクエストイベント
    
    文字起こし結果を翻訳するためのリクエスト。
    """
    event_type: Literal['translation_request']
    text: str                    # 翻訳対象テキスト
    source_id: str              # ソース識別子
    source_language: str        # ソース言語コード
    target_language: str        # ターゲット言語コード
    timestamp: float            # Unix timestamp
    metadata: Optional[Dict[str, Any]]  # 追加メタデータ


class TranslationResultEventDict(TypedDict, total=False):
    """
    翻訳結果イベント
    
    翻訳処理の完了結果。
    """
    event_type: Literal['translation_result']
    original_text: str          # 原文
    translated_text: str        # 翻訳文
    source_id: str              # ソース識別子
    source_language: str        # ソース言語
    target_language: str        # ターゲット言語
    timestamp: float            # Unix timestamp
    confidence: Optional[float]  # 翻訳信頼度
    metadata: Optional[Dict[str, Any]]


class SubtitleEventDict(TypedDict, total=False):
    """
    字幕送信イベント
    
    OBSやVRChatへの字幕送信用イベント。
    """
    event_type: Literal['subtitle']
    text: str                   # 表示テキスト
    source_id: str              # ソース識別子
    destination: str            # 送信先: 'obs' or 'vrchat'
    is_translated: bool         # 翻訳済みフラグ
    original_text: Optional[str]  # 原文（翻訳時のみ）
    timestamp: float            # Unix timestamp
    display_params: Optional[Dict[str, Any]]  # 表示パラメータ


# === 統合型定義 ===
UiEventDict = Union[TranscriptionEventDict, StatusEventDict, ErrorEventDict]

# Phase 4: 拡張イベント型を含む統一型
ExtendedEventDict = Union[
    TranscriptionEventDict,
    StatusEventDict,
    ErrorEventDict,
    TranslationRequestEventDict,
    TranslationResultEventDict,
    SubtitleEventDict
]


# === 互換性のための旧型定義（段階的削除予定） ===
class MinimalTranscriptionDict(TypedDict):
    """エンジンレベルの最小限の辞書（後方互換性）"""
    text: str
    confidence: float


# === バリデーション ===

def _rehydrate_event_dict(data: dict) -> Optional[UiEventDict]:
    """
    event_type付き辞書から欠損フィールドを補完し、標準イベントに再構築する。
    """
    event_type = data.get('event_type')

    if event_type == 'transcription':
        metadata = data.get('metadata') or {}
        if not isinstance(metadata, dict):
            metadata = {'raw_metadata': metadata}
        else:
            metadata = dict(metadata)  # defensive copy

        status = data.get('status')
        if status is not None and 'status' not in metadata:
            metadata['status'] = status

        is_final: Optional[bool] = data.get('is_final')
        if is_final is None:
            is_final = metadata.get('is_final', False)

        text = (
            data.get('text')
            or metadata.get('text')
            or data.get('display_text')
            or status
            or ""
        )
        if not text:
            return None

        kwargs: Dict[str, Any] = {
            'confidence': data.get('confidence'),
            'language': data.get('language'),
            'phase': data.get('phase'),
            'display_text': data.get('display_text'),
            'vad_state': data.get('vad_state'),
            'speech_probability': data.get('speech_probability'),
            'audio_quality': data.get('audio_quality'),
            'noise_level': data.get('noise_level'),
        }

        timestamp = data.get('timestamp') or metadata.get('timestamp')
        if metadata:
            kwargs['metadata'] = metadata

        normalized = create_transcription_event(
            text=text,
            source_id=data.get('source_id', 'unknown'),
            is_final=bool(is_final),
            timestamp=timestamp,
            **{k: v for k, v in kwargs.items() if v is not None}
        )
        return normalized

    return None


# === バリデーション ===

def validate_event_dict(data: dict) -> bool:
    """
    辞書が有効なイベント形式かを検証
    
    Args:
        data: 検証する辞書
        
    Returns:
        有効な場合 True
    """
    if not isinstance(data, dict):
        return False
        
    event_type = data.get('event_type')
    
    # 既知のイベントタイプのみを有効とする
    valid_event_types = ['transcription', 'status', 'error', 
                        'translation_request', 'translation_result', 'subtitle', 
                        None]  # Noneは後方互換性
    
    if event_type not in valid_event_types:
        # 未知のイベントタイプは無効
        return False
    
    if event_type == 'transcription':
        required = ['text', 'source_id', 'is_final', 'timestamp', 'event_type']
    elif event_type == 'status':
        required = ['event_type', 'status_code', 'message', 'timestamp', 'source_id']
    elif event_type == 'error':
        required = ['event_type', 'error_code', 'message', 'timestamp', 'source_id']
    elif event_type == 'translation_request':
        required = ['event_type', 'text', 'source_id', 'source_language', 
                   'target_language', 'timestamp']
    elif event_type == 'translation_result':
        required = ['event_type', 'original_text', 'translated_text', 'source_id',
                   'source_language', 'target_language', 'timestamp']
    elif event_type == 'subtitle':
        required = ['event_type', 'text', 'source_id', 'destination', 
                   'is_translated', 'timestamp']
    else:
        # event_typeが無い場合（後方互換性）
        # transcriptionとして扱う
        required = ['text', 'source_id']
    
    return all(field in data for field in required)


# === 特化型バリデーション ===

def validate_translation_event(data: dict) -> bool:
    """
    翻訳関連イベントの検証
    
    Args:
        data: 検証する辞書
        
    Returns:
        有効な翻訳イベントの場合 True
    """
    event_type = data.get('event_type')
    if event_type not in ['translation_request', 'translation_result']:
        return False
    
    # 言語コードの妥当性チェック
    if event_type == 'translation_request':
        source_lang = data.get('source_language')
        target_lang = data.get('target_language')
        if source_lang == target_lang:
            logger.warning(f"Same source and target language: {source_lang}")
            return False
    
    return validate_event_dict(data)


def validate_subtitle_event(data: dict) -> bool:
    """
    字幕イベントの検証
    
    Args:
        data: 検証する辞書
        
    Returns:
        有効な字幕イベントの場合 True
    """
    if data.get('event_type') != 'subtitle':
        return False
    
    # 送信先の妥当性チェック
    destination = data.get('destination')
    if destination not in ['obs', 'vrchat']:
        logger.warning(f"Invalid subtitle destination: {destination}")
        return False
    
    # VRChat特有のバリデーション
    if destination == 'vrchat':
        display_params = data.get('display_params', {})
        if display_params.get('max_lines', 1) > 1:
            logger.warning("VRChat only supports single line display")
    
    return validate_event_dict(data)


# === ヘルパー関数 ===

def create_transcription_event(
    text: str,
    source_id: str,
    is_final: bool = True,
    timestamp: Optional[float] = None,
    **kwargs
) -> TranscriptionEventDict:
    """
    標準的な TranscriptionEventDict を作成
    
    Args:
        text: 認識されたテキスト
        source_id: ソース識別子
        is_final: 確定済みフラグ
        timestamp: タイムスタンプ（省略時は現在時刻）
        **kwargs: その他の任意フィールド
        
    Returns:
        TranscriptionEventDict形式の辞書
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    result: TranscriptionEventDict = {
        'event_type': 'transcription',
        'text': text,
        'source_id': source_id,
        'is_final': is_final,
        'timestamp': timestamp
    }
    
    # 任意フィールドを追加
    valid_optional = ['confidence', 'language', 'phase', 'display_text',
                     'vad_state', 'speech_probability', 'audio_quality',
                     'noise_level', 'metadata']
    for key, value in kwargs.items():
        if key in valid_optional and value is not None:
            result[key] = value
    
    # phaseのデフォルト値設定
    if 'phase' not in result:
        result['phase'] = 'final' if is_final else 'interim'
    
    return result


def create_status_event(
    status_code: str,
    message: str,
    source_id: str,
    timestamp: Optional[float] = None,
    **kwargs
) -> StatusEventDict:
    """
    標準的な StatusEventDict を作成
    
    Args:
        status_code: ステータスコード
        message: ステータスメッセージ
        source_id: ソース識別子
        timestamp: タイムスタンプ（省略時は現在時刻）
        **kwargs: その他の任意フィールド
        
    Returns:
        StatusEventDict形式の辞書
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    result: StatusEventDict = {
        'event_type': 'status',
        'status_code': status_code,
        'message': message,
        'source_id': source_id,
        'timestamp': timestamp
    }
    
    # 任意フィールドを追加
    if 'phase' in kwargs:
        result['phase'] = kwargs['phase']
    if 'metadata' in kwargs:
        result['metadata'] = kwargs['metadata']
    
    return result


def create_error_event(
    error_code: str,
    message: str,
    source_id: str,
    timestamp: Optional[float] = None,
    **kwargs
) -> ErrorEventDict:
    """
    標準的な ErrorEventDict を作成
    
    Args:
        error_code: エラーコード
        message: エラーメッセージ
        source_id: ソース識別子
        timestamp: タイムスタンプ（省略時は現在時刻）
        **kwargs: その他の任意フィールド
        
    Returns:
        ErrorEventDict形式の辞書
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    result: ErrorEventDict = {
        'event_type': 'error',
        'error_code': error_code,
        'message': message,
        'source_id': source_id,
        'timestamp': timestamp
    }
    
    # 任意フィールドを追加
    if 'error_details' in kwargs:
        result['error_details'] = kwargs['error_details']
    if 'metadata' in kwargs:
        result['metadata'] = kwargs['metadata']
    
    return result


def create_translation_request_event(
    text: str,
    source_id: str,
    source_language: str,
    target_language: str,
    timestamp: Optional[float] = None,
    **kwargs
) -> TranslationRequestEventDict:
    """
    翻訳リクエストイベントを作成
    
    Args:
        text: 翻訳対象テキスト
        source_id: ソース識別子
        source_language: ソース言語コード
        target_language: ターゲット言語コード
        timestamp: タイムスタンプ（省略時は現在時刻）
        **kwargs: その他の任意フィールド
        
    Returns:
        TranslationRequestEventDict形式の辞書
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    result: TranslationRequestEventDict = {
        'event_type': 'translation_request',
        'text': text,
        'source_id': source_id,
        'source_language': source_language,
        'target_language': target_language,
        'timestamp': timestamp
    }
    
    # メタデータを追加
    if 'metadata' in kwargs:
        result['metadata'] = kwargs['metadata']

    return result


def create_translation_result_event(
    original_text: str,
    translated_text: str,
    source_id: str,
    source_language: str,
    target_language: str,
    timestamp: Optional[float] = None,
    **kwargs
) -> TranslationResultEventDict:
    """
    翻訳結果イベントを作成
    
    Args:
        original_text: 原文
        translated_text: 翻訳文
        source_id: ソース識別子
        source_language: ソース言語コード
        target_language: ターゲット言語コード
        timestamp: タイムスタンプ（省略時は現在時刻）
        **kwargs: その他の任意フィールド
        
    Returns:
        TranslationResultEventDict形式の辞書
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    result: TranslationResultEventDict = {
        'event_type': 'translation_result',
        'original_text': original_text,
        'translated_text': translated_text,
        'source_id': source_id,
        'source_language': source_language,
        'target_language': target_language,
        'timestamp': timestamp
    }
    
    # 任意フィールドを追加
    if 'confidence' in kwargs:
        result['confidence'] = kwargs['confidence']
    if 'metadata' in kwargs:
        result['metadata'] = kwargs['metadata']

    return result


def create_subtitle_event(
    text: str,
    source_id: str,
    destination: str,
    is_translated: bool,
    timestamp: Optional[float] = None,
    **kwargs
) -> SubtitleEventDict:
    """
    字幕イベントを作成
    
    Args:
        text: 表示テキスト
        source_id: ソース識別子
        destination: 送信先 ('obs' or 'vrchat')
        is_translated: 翻訳済みフラグ
        timestamp: タイムスタンプ（省略時は現在時刻）
        **kwargs: その他の任意フィールド
        
    Returns:
        SubtitleEventDict形式の辞書
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    # 送信先の検証
    if destination not in ['obs', 'vrchat']:
        raise ValueError(f"Invalid destination: {destination}")
    
    result: SubtitleEventDict = {
        'event_type': 'subtitle',
        'text': text,
        'source_id': source_id,
        'destination': destination,
        'is_translated': is_translated,
        'timestamp': timestamp
    }
    
    # 任意フィールドを追加
    if 'original_text' in kwargs:
        result['original_text'] = kwargs['original_text']
    if 'display_params' in kwargs:
        result['display_params'] = kwargs['display_params']
    if 'is_final' in kwargs:
        result['is_final'] = kwargs['is_final']
    
    return result


# === 型変換ヘルパー（後方互換性） ===

def normalize_to_event_dict(data: dict) -> Optional[UiEventDict]:
    """
    既存の辞書形式を標準イベント形式に正規化
    
    Phase 3改善: より堅牢な正規化とデフォルト値の設定
    
    Args:
        data: 変換元の辞書
        
    Returns:
        正規化されたイベント辞書、変換できない場合はNone
    """
    # すでにevent_typeがある場合はそのまま返す
    if 'event_type' in data:
        if validate_event_dict(data):
            return data
        rehydrated = _rehydrate_event_dict(data)
        if rehydrated and validate_event_dict(rehydrated):
            return rehydrated
        logger.warning(f"Invalid event dict with event_type: {data}")
        return None
    
    # event_typeがない場合は、内容から推測して変換
    # textフィールドがあればtranscriptionイベントとして扱う
    if 'text' in data and 'source_id' in data:
        return create_transcription_event(
            text=data['text'],
            source_id=data['source_id'],
            is_final=data.get('is_final', True),
            timestamp=data.get('timestamp'),
            confidence=data.get('confidence'),
            language=data.get('language'),
            phase=data.get('phase'),  # phaseも継承
            vad_state=data.get('vad_state'),
            speech_probability=data.get('speech_probability'),
            audio_quality=data.get('audio_quality'),
            noise_level=data.get('noise_level'),
            display_text=data.get('display_text'),
            metadata=data.get('metadata')
        )
    
    # status_codeとmessageがあればstatusイベント
    if 'status_code' in data and 'message' in data:
        return create_status_event(
            status_code=data['status_code'],
            message=data['message'],
            source_id=data.get('source_id', 'unknown'),
            timestamp=data.get('timestamp'),
            phase=data.get('phase'),
            metadata=data.get('metadata')
        )
    
    # error_codeとmessageがあればerrorイベント
    if 'error_code' in data and 'message' in data:
        return create_error_event(
            error_code=data['error_code'],
            message=data['message'],
            source_id=data.get('source_id', 'unknown'),
            timestamp=data.get('timestamp'),
            error_details=data.get('error_details'),
            metadata=data.get('metadata')
        )
    
    logger.warning(f"Cannot normalize dict to event format: {data}")
    return None


# === デバッグ用ヘルパー ===

def get_event_type_name(event: UiEventDict) -> str:
    """
    イベントのタイプ名を取得
    
    Args:
        event: イベント辞書
        
    Returns:
        イベントタイプ名
    """
    return event.get('event_type', 'unknown')


def format_event_summary(event: UiEventDict) -> str:
    """
    イベントの概要を文字列形式で取得
    
    Args:
        event: イベント辞書
        
    Returns:
        概要文字列
    """
    event_type = get_event_type_name(event)
    source_id = event.get('source_id', 'unknown')
    
    if event_type == 'transcription':
        text = event.get('text', '')
        is_final = event.get('is_final', False)
        return f"[{source_id}] {'Final' if is_final else 'Interim'}: {text[:50]}..."
    elif event_type == 'status':
        status_code = event.get('status_code', 'unknown')
        message = event.get('message', '')
        return f"[{source_id}] Status {status_code}: {message}"
    elif event_type == 'error':
        error_code = event.get('error_code', 'unknown')
        message = event.get('message', '')
        return f"[{source_id}] Error {error_code}: {message}"
    else:
        return f"[{source_id}] Unknown event type: {event_type}"
