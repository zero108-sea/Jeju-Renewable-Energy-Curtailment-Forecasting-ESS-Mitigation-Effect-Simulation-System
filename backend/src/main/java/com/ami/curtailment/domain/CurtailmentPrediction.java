package com.ami.curtailment.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * AI 서버(POST /predict)로부터 받은 출력제어 확률 예측 결과 저장.
 *
 * 필드는 AI 서버 실제 응답(app/schemas.py) 기준이다.
 *
 * - curtailmentProbability: sigmoid 확률 보정이 적용된 값
 * - note: 모든 경로(SOLAR, WIND, WIND+수요)에서 존재할 수 있음. TEXT 컬럼 사용
 * - ESS 흡수율 계산에는 curtailmentMwh(예측값)를 쓰지 말 것. 실측 제어량 기반만 사용
 *   (예측값 사용 시 실측 36.8%가 78%로 부풀려짐)
 * - 0.03 같은 고정 임계값 사용 금지. modelUsed별로 operationalThreshold가 다르므로
 *   응답값을 그대로 저장해야 함
 * - operationalThresholdReliable=false이면 등급(낮음/보통/높음) 확정 표시 대신
 *   "상위 5%" 방식으로 대체 표시할 것
 * - modelTrainedAt: AI 서버 응답의 model_trained_at. 재학습 전후 예측을 구분하는 유일한 단서
 */
@Entity
@Table(name = "curtailment_predictions")
@Getter
@Setter
@NoArgsConstructor
public class CurtailmentPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "region_id", nullable = false)
    private Region region;

    @Column(nullable = false)
    private LocalDateTime targetHour; // 예측 대상 시각

    @Column(nullable = false)
    private double curtailmentProbability; // 출력제어 확률 (0~1, sigmoid 보정값)

    @Column(length = 60)
    private String modelUsed; // classifier_{solar,solar_demand,wind,wind_demand,wind_demand_crossp}_calibrated_sigmoid

    @Column(length = 40)
    private String modelTrainedAt; // AI 서버 model_trained_at (ISO 문자열). 재학습 전후 구분용

    private Double operationalThreshold; // 모델별 운영 임계값 (0.02~0.43, 최대 20배 차이). 0.03 고정 사용 금지

    private Boolean operationalThresholdReliable; // false면 임계값 확정 불가. 상위 5%로만 표시

    private double generationForecastMwh; // 컨버터 산출 발전량 예측치

    private Double curtailmentMwh; // 예상 출력제어량 (nullable). SOLAR는 항상 null, WIND만 값 존재

    @Column(columnDefinition = "TEXT")
    private String note; // 경고문이 길어질 수 있어 length 제한(255) 대신 TEXT 사용

    @Column(nullable = false)
    private LocalDateTime predictedAt; // 예측이 생성된 시각
}
