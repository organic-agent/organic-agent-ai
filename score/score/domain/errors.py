"""사진 단위 결정적 실패 표시(#85, wes V15) — `photo_analysis.error` 에 쓰는 값.

wes 는 값을 해석하지 않고 "실패했다"로만 본다(기대 장수에서 뺀다) — 짧은 코드로 둔다. Lambda 폴백(job)과 GPU 워커(worker)가
같은 값을 쓴다.
"""

#: 미리보기가 S3 에 없다(404). 다시 받아도 안 되는 실패라 그 장만 뺀다.
PREVIEW_MISSING = "PREVIEW_MISSING"
#: 디코드·추론 실패(pipeline 의 failed).
SCORE_FAILED = "SCORE_FAILED"
