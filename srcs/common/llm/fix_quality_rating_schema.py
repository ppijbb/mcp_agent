"""
QualityRating Schema Fix

mcp_agent 라이브러리의 QualityRating enum validation 오류를 수정하는 monkey patch
LLM이 'EXCELLENT', 'GOOD', 'FAIR', 'POOR' 문자열을 반환하지만
EvaluationResult는 정수 enum (0, 1, 2, 3)을 기대하는 문제 해결

해결 방법:
1. JSON 스키마를 문자열 enum으로 수정 (LLM이 문자열 반환)
2. EvaluationResult에 validator 추가하여 문자열을 정수 enum으로 변환
"""

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_patched = False


def patch_quality_rating_schema():
    """
    QualityRating enum의 validation 오류 수정
    1. JSON 스키마를 문자열 enum으로 수정
    2. EvaluationResult에 validator 추가하여 문자열을 정수 enum으로 변환
    """
    global _patched
    if _patched:
        return

    try:
        from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import EvaluationResult, QualityRating

        # 1. JSON 스키마를 문자열 enum으로 수정
        original_model_json_schema = EvaluationResult.model_json_schema

        @classmethod
        def patched_model_json_schema(cls, by_alias: bool = True, ref_template: str = "#/$defs/{model}", **kwargs):
            """QualityRating enum을 문자열로 변환한 스키마 반환"""
            schema = original_model_json_schema(by_alias=by_alias, ref_template=ref_template, **kwargs)

            # $defs의 QualityRating enum 정의를 문자열로 수정
            if '$defs' in schema and 'QualityRating' in schema['$defs']:
                quality_rating_def = schema['$defs']['QualityRating']
                if 'enum' in quality_rating_def and isinstance(quality_rating_def['enum'], list):
                    enum_values = quality_rating_def['enum']
                    # 정수값인지 확인 (0, 1, 2, 3)
                    if all(isinstance(v, int) for v in enum_values):
                        # QualityRating enum의 name을 사용 (EXCELLENT, GOOD, FAIR, POOR)
                        quality_rating_def['enum'] = [rating.name for rating in QualityRating]
                        quality_rating_def['type'] = 'string'
                        logger.debug(f"Fixed QualityRating enum in $defs: {quality_rating_def['enum']}")

            return schema

        EvaluationResult.model_json_schema = patched_model_json_schema

        # 2. EvaluationResult에 validator 추가하여 문자열을 정수 enum으로 변환
        def convert_rating_string_to_enum(v: Union[str, int, QualityRating]) -> QualityRating:
            """문자열 rating을 정수 enum으로 변환"""
            if isinstance(v, QualityRating):
                return v
            elif isinstance(v, int):
                # 정수값을 직접 QualityRating enum으로 변환
                try:
                    return QualityRating(v)
                except ValueError:
                    # 범위를 벗어난 정수값 처리
                    logger.warning(f"Invalid QualityRating integer value: {v}, defaulting to GOOD")
                    return QualityRating.GOOD
            elif isinstance(v, str):
                # 문자열을 QualityRating enum으로 변환
                v_upper = v.upper().strip()
                rating_map = {
                    'EXCELLENT': QualityRating.EXCELLENT,
                    'GOOD': QualityRating.GOOD,
                    'FAIR': QualityRating.FAIR,
                    'POOR': QualityRating.POOR,
                }
                if v_upper in rating_map:
                    return rating_map[v_upper]
                else:
                    logger.warning(f"Invalid QualityRating string value: {v}, defaulting to GOOD")
                    return QualityRating.GOOD
            else:
                logger.warning(f"Unexpected QualityRating type: {type(v)}, defaulting to GOOD")
                return QualityRating.GOOD

        # Pydantic v2에서는 model_validate를 monkey patch
        if not hasattr(EvaluationResult, '__quality_rating_validator_patched__'):
            EvaluationResult.__quality_rating_validator_patched__ = True

            # model_validate를 monkey patch
            original_model_validate = EvaluationResult.model_validate

            @classmethod
            def patched_model_validate(cls, obj, *args, **kwargs):
                """model_validate에서 rating 문자열 변환"""
                if isinstance(obj, dict) and 'rating' in obj:
                    if isinstance(obj['rating'], str):
                        obj = obj.copy()
                        obj['rating'] = convert_rating_string_to_enum(obj['rating'])
                return original_model_validate(obj, *args, **kwargs)

            EvaluationResult.model_validate = patched_model_validate

            # __init__도 patch
            original_init = EvaluationResult.__init__

            def patched_init(self, *args, **kwargs):
                # rating이 문자열인 경우 변환
                if 'rating' in kwargs and isinstance(kwargs['rating'], str):
                    kwargs['rating'] = convert_rating_string_to_enum(kwargs['rating'])
                return original_init(self, *args, **kwargs)

            EvaluationResult.__init__ = patched_init

        _patched = True
        logger.info("QualityRating schema and validator patch applied successfully")

    except Exception as e:
        logger.warning(f"Failed to patch QualityRating schema: {e}", exc_info=True)
        _patched = False


# 모듈 import 시 자동으로 patch 적용
try:
    patch_quality_rating_schema()
except Exception as e:
    logger.warning(f"Auto-patch failed: {e}")
