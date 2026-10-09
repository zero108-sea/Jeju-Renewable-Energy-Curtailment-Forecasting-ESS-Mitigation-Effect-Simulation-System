# 제주 출력제어 예측 프로젝트 — Backend (Spring Boot)

담당: 이성헌 | 스택: Java 21, Spring Boot 3.3, PostgreSQL

**주의**: DTO/엔드포인트는 확정서 v1.0(9/8, camelCase, `/api/v1/...`)이 아니라
지호님이 실제로 구현·배포한 AI 서버(app/schemas.py, snake_case, `/predict`,
`/ess/simulate`) 기준으로 맞춰져 있습니다. 두 문서가 다르면 이 README와
AI 서버 저장소 쪽 README를 기준으로 삼으세요.

## 폴더 구조

```
domain/       Region, EnergySource(SOLAR/WIND), PowerGeneration,
              CurtailmentPrediction, EssSimulationResult
repository/   각 Entity의 JPA Repository
controller/   RegionController, CurtailmentPredictionController(POST /api/predictions),
              EssSimulationController(POST /api/ess-simulations)
service/      AiClientService(AI 서버 실제 경로 /predict, /ess/simulate 호출),
              EssSimulationService(호출+저장, 계산 자체는 AI 서버가 수행)
dto/          PredictionRequest/Response, WeatherHour, HourlyPrediction,
              EssSimulateRequest/Response, ErrorResponse
config/       SecurityConfig, WebClientConfig, GlobalExceptionHandler
```

## AI 서버 실제 스펙 반영 내역 (이번 수정)

- 엔드포인트: `/predict`, `/ess/simulate` (AI 서버 실제 경로, `/api/v1/...` 아님)
- 필드명: snake_case로 통일 (`energy_type`, `target_date`, `wind_speed`,
  `solar_rad`, `curtailment_probability`, `expected_curtailment_mwh` 등)
- **시간 인덱스가 1~24** — Backend는 원본 그대로 1~24를 주고받으며 가공하지 않음(지호님 확인).
  저장 시에만 `CurtailmentPredictionController`에서 24시를 다음날 00:00으로 변환 (1~23시는
  그날 해당 시각 그대로). 이전 버전의 "-1 일괄 보정"은 잘못된 가정이었어서 정정함
- ESS 계산은 **AI 서버가 수행** — Backend는 `hourly_curtailment_mwh`, `rated_power_mw`,
  `method`를 보내고 `total_absorbed_mwh`, `absorption_rate`를 받아 저장만 함
  (이전 버전은 Backend에서 직접 min() 계산했으나 실제 구현과 역할이 달라 정정)
- `expected_curtailment_mwh`는 **WIND만** 값 존재, SOLAR는 항상 null +
  `note` 필드로 사유 안내 — Frontend에서 SOLAR 표시 시 note 문구 필수 노출
- 검증 책임: Backend는 필드 존재 여부만 확인, weather 24개·중복 검증은
  AI 서버가 전담 (위반 시 422 `INVALID_HOUR_SET`, 기상값 범위 초과 시 422 `OUT_OF_RANGE`,
  `GlobalExceptionHandler`가 그대로 릴레이)
- `riskLevel`, `modelVersion` 필드 제거 (실제 응답에 없음)
- 헬스체크 엔드포인트 제거 (AI 서버에 문서화되어 있지 않음)

## 실행 방법

1. PostgreSQL에 DB 생성
   ```sql
   CREATE DATABASE curtailment_db;
   ```
2. `application.yml`에서 비밀번호 설정, `ai-server.base-url`이 `localhost:8001`인지 확인
   (AI 서버 실행법: `uvicorn app.main:app --reload --port 8001`)
3. IntelliJ에서 실행 (JDK 21)
4. AI 서버 저장소를 클론해서 먼저 띄운 뒤 연동 테스트

## 지금 상태

- [x] Entity/Repository/Controller/DTO — AI 서버 실제 구현체 기준 재작성 완료
- [x] AI 서버 에러(422 등) pass-through 처리
- [ ] AI 서버 로컬 기동 + 실제 연동 테스트 (다음 단계)
- [ ] 로컬 실행 테스트 (PostgreSQL 연결 후 API 테스트)
