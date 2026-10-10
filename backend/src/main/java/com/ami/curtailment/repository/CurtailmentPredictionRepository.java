package com.ami.curtailment.repository;

import com.ami.curtailment.domain.CurtailmentPrediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface CurtailmentPredictionRepository extends JpaRepository<CurtailmentPrediction, Long> {
    // 전체 이력 (같은 시각의 예측이 재호출·재학습마다 여러 건 쌓여 있다)
    List<CurtailmentPrediction> findByRegionIdOrderByTargetHourDesc(Long regionId);

    // 시각(targetHour)별로 가장 나중에 저장된 1건만. POST는 호출마다 24행을 새로 저장하므로
    // (재학습 전후 예측을 비교하려면 이력을 지우면 안 된다) 조회에서 중복을 걷어낸다.
    // 최신 판정은 predictedAt이 아니라 id(IDENTITY, 단조 증가)로 한다 - 시각 동률이 없다.
    @Query("""
            select p from CurtailmentPrediction p
            where p.region.id = :regionId
              and p.id = (select max(p2.id) from CurtailmentPrediction p2
                          where p2.region.id = p.region.id and p2.targetHour = p.targetHour)
            order by p.targetHour desc
            """)
    List<CurtailmentPrediction> findLatestByRegionId(@Param("regionId") Long regionId);
}
