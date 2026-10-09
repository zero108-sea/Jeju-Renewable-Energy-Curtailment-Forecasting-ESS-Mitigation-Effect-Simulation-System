package com.ami.curtailment.dto;

import com.ami.curtailment.domain.CurtailmentPrediction;
import lombok.Getter;

import java.time.LocalDateTime;

/**
 * GET /api/predictions/{regionId} 응답 전용 DTO.
 *
 * 통신규격 v1.1 07장 버그 반영 (2026-10-01): CurtailmentPrediction 엔티티를 그대로
 * 반환하면 region이 Hibernate 지연 로딩 프록시로 남아 Jackson 직렬화 시 500
 * (ByteBuddyInterceptor 직렬화 실패)이 난다. POST 저장 직후의 실제 인스턴스는 문제없지만,
 * 조회(GET)는 항상 프록시를 반환하므로 이 DTO로 변환해서 내려줄 것. 지호님 권장 방법 채택.
 * 부수 효과로 엔티티 변경이 API 응답 계약을 깨지 않게 한다.
 */
@Getter
public class CurtailmentPredictionView {
    private final Long id;
    private final String regionName;
    private final LocalDateTime targetHour;
    private final double curtailmentProbability;
    private final String modelUsed;
    private final String modelTrainedAt; // 2026-10-09 추가: 재학습 전후 예측 구분용
    private final Double operationalThreshold;
    private final Boolean operationalThresholdReliable;
    private final double generationForecastMwh;
    private final Double curtailmentMwh;
    private final String note;
    private final LocalDateTime predictedAt;

    public CurtailmentPredictionView(CurtailmentPrediction entity) {
        this.id = entity.getId();
        this.regionName = entity.getRegion().getName(); // 트랜잭션 안에서 프록시를 초기화해 값만 꺼냄
        this.targetHour = entity.getTargetHour();
        this.curtailmentProbability = entity.getCurtailmentProbability();
        this.modelUsed = entity.getModelUsed();
        this.modelTrainedAt = entity.getModelTrainedAt();
        this.operationalThreshold = entity.getOperationalThreshold();
        this.operationalThresholdReliable = entity.getOperationalThresholdReliable();
        this.generationForecastMwh = entity.getGenerationForecastMwh();
        this.curtailmentMwh = entity.getCurtailmentMwh();
        this.note = entity.getNote();
        this.predictedAt = entity.getPredictedAt();
    }
}
