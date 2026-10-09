package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * AI 서버 POST /predict 응답. app/schemas.py 실제 필드명 그대로 사용한다.
 *
 * [2026-09-24 변경] note는 더 이상 태양광 전용이 아니다. 풍력 + demand_forecast_mw 경로에서도
 * ESS 경고문("expected_curtailment_mwh는 기댓값이므로 /ess/simulate에 넣지 말 것")이 올 수 있다.
 * Frontend가 "태양광일 때만 note를 본다"는 전제로 분기하면 이 경고가 화면에 노출되지 않는다.
 * energy_type과 무관하게 note가 비어있지 않으면 노출할 것 (AI 서버 README 명시).
 */
@Getter
@Setter
@NoArgsConstructor
public class PredictionResponse {
    private String energy_type;
    private String region;
    private String target_date;
    private List<HourlyPrediction> hourly;

    // curtailment_probability를 산출한 보정 분류모델 이름. 값을 하드코딩해 분기하지 말 것.
    // 2026-09-26 현재 5경로: classifier_{solar,solar_demand,wind,wind_demand,wind_demand_crossp}_calibrated_sigmoid
    private String model_used;

    // [2026-10-06 추가] 모델 학습 시각 (ISO 문자열, 예: 2026-09-27T06:22:32+00:00).
    // model_used는 경로 이름이라 재학습해도 바뀌지 않으므로, 재학습 전후 예측을 구분하는 유일한 단서다.
    private String model_trained_at;

    // model_used별로 확정된 '제어 발생 경보' 임계값.
    // curtailment_probability >= 이 값이면 경보로 취급한다.
    // 0.03 같은 값을 하드코딩하지 말 것. 경로마다 다르며 현재 0.02~0.43으로 20배 차이가 난다.
    private Double operational_threshold;

    // false면 임계값 확정 구간의 양성 표본이 부족했다는 뜻이다. 이 경우 임계값 판정을 하지 말고
    // 등급(상위 5%)으로만 표시할 것. note에도 안내가 들어온다.
    private Boolean operational_threshold_reliable;

    // 존재 조건: (1) 태양광 - expected_curtailment_mwh가 null인 이유,
    //           (2) 풍력 + 수요예측 생략 - 수요 미포함 모델로 전환됐다는 안내,
    //           (3) 풍력 + 수요예측 포함 - ESS 계산에 쓰지 말라는 경고
    private String note;
}
