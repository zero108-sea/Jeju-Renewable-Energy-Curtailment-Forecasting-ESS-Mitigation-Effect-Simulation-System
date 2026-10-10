package com.ami.curtailment.controller;

import com.ami.curtailment.domain.CurtailmentPrediction;
import com.ami.curtailment.domain.Region;
import com.ami.curtailment.dto.CurtailmentPredictionView;
import com.ami.curtailment.dto.HourlyPrediction;
import com.ami.curtailment.dto.PredictionRequest;
import com.ami.curtailment.dto.PredictionResponse;
import com.ami.curtailment.repository.CurtailmentPredictionRepository;
import com.ami.curtailment.repository.RegionRepository;
import com.ami.curtailment.service.AiClientService;
import lombok.RequiredArgsConstructor;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 외부 노출 API: 출력제어 예측 요청 및 결과 조회.
 * AI 서버 실제 엔드포인트(POST /predict) 기준 - 확정서 v1.0의 /api/v1/predictions/daily가 아님.
 */
@RestController
@RequestMapping("/api/predictions")
@RequiredArgsConstructor
public class CurtailmentPredictionController {

    private final AiClientService aiClientService;
    private final RegionRepository regionRepository;
    private final CurtailmentPredictionRepository curtailmentPredictionRepository;

    // 지역별 저장된 예측 결과 조회 (DB 조회, AI 서버 호출 아님)
    // 기본은 시각별 최신 1건. 같은 날짜를 여러 번 예측하면 24행씩 계속 쌓이므로(재학습 전후
    // 비교용 이력 보존) 조회에서 중복을 걷어낸다. 전체 이력은 ?history=true.
    // 통신규격 v1.1 07장 버그 수정: CurtailmentPrediction 엔티티를 직접 반환하면 region이
    // Hibernate 지연 로딩 프록시로 남아 Jackson 직렬화가 500으로 실패함 - 응답 전용 DTO로 변환
    @Transactional(readOnly = true)
    @GetMapping("/{regionId}")
    public List<CurtailmentPredictionView> getPredictions(
            @PathVariable Long regionId,
            @RequestParam(name = "history", defaultValue = "false") boolean history) {
        List<CurtailmentPrediction> rows = history
                ? curtailmentPredictionRepository.findByRegionIdOrderByTargetHourDesc(regionId)
                : curtailmentPredictionRepository.findLatestByRegionId(regionId);
        return rows.stream()
                .map(CurtailmentPredictionView::new)
                .toList();
    }

    /**
     * 하루치 1~24시 원본 데이터셋 관례 예측. AI 서버 POST /predict 호출 후 24개 결과를 각각 저장.
     * weather 요청 필드의 hour 값도 노출되는 쪽(Frontend 포함)이 1~24 그대로 채워서 보내면 되고,
     * Backend는 이 값을 가공하지 않고 그대로 AI 서버에 전달한다 (지호님 확인 사항).
     */
    @PostMapping
    public List<CurtailmentPrediction> predict(@RequestBody PredictionRequest request) {
        Region region = regionRepository.findByName(request.getRegion());
        if (region == null) {
            // 미등록 지역이면 AI 서버를 부르기 전에 400으로 끝낸다 (GlobalExceptionHandler가 INVALID_REQUEST로 변환).
            // 이전에는 region이 null인 채로 저장을 시도해 DB 제약 위반(500)으로 터졌다.
            throw new IllegalArgumentException("등록되지 않은 지역입니다: " + request.getRegion());
        }
        LocalDate targetDate = LocalDate.parse(request.getTarget_date());

        PredictionResponse response = aiClientService.predict(request);

        return response.getHourly().stream()
                .map(hp -> saveOne(region, targetDate, response, hp))
                .toList();
    }

    private CurtailmentPrediction saveOne(Region region, LocalDate targetDate,
                                          PredictionResponse response, HourlyPrediction hp) {
        CurtailmentPrediction prediction = new CurtailmentPrediction();
        prediction.setRegion(region);
        // 원본 데이터셋 관례: 1~23시는 그날 해당 시각, 24시는 "자정"=다음날 00:00.
        // (지호님 확인: "현재 데이터셋 파싱 시에 자정 24시를 0시로 바꾸고 있는데, AI에서 이걸
        //  처리하므로 Backend는 원본 1~24 그대로 보내면 됨". 응답 저장 시에도 같은 관례 적용)
        LocalDateTime hourStart = (hp.getHour() == 24)
                ? targetDate.plusDays(1).atStartOfDay()
                : targetDate.atTime(hp.getHour(), 0);
        prediction.setTargetHour(hourStart);
        prediction.setCurtailmentProbability(hp.getCurtailment_probability());
        prediction.setGenerationForecastMwh(hp.getGeneration_forecast_mwh());
        prediction.setCurtailmentMwh(hp.getExpected_curtailment_mwh());
        prediction.setNote(response.getNote());

        // 2026-09-26 추가분: 모델 식별값 + 운영 임계값 (0.03 고정 금지, 응답값 그대로 저장)
        prediction.setModelUsed(response.getModel_used());
        // 2026-10-09 추가: 재학습 전후 예측을 구분하기 위한 모델 학습 시각
        prediction.setModelTrainedAt(response.getModel_trained_at());
        prediction.setOperationalThreshold(response.getOperational_threshold());
        prediction.setOperationalThresholdReliable(response.getOperational_threshold_reliable());

        prediction.setPredictedAt(LocalDateTime.now());

        return curtailmentPredictionRepository.save(prediction);
    }
}