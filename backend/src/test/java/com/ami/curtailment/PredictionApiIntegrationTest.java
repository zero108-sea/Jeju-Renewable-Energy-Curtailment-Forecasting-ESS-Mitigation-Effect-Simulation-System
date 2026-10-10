package com.ami.curtailment;

import com.jayway.jsonpath.JsonPath;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.everyItem;
import static org.hamcrest.Matchers.is;
import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Backend 통합 테스트: Controller -> WebClient -> (가짜 AI 서버) -> JPA(H2) -> GET 전 구간.
 *
 * 가짜 AI 서버는 JDK 내장 HttpServer로 띄운다(추가 의존성 없음). 진짜 AI 서버와 같은 snake_case JSON을
 * 돌려주므로 WebClient의 JSON 매핑과 GlobalExceptionHandler의 에러 전달까지 실제 코드로 검증된다.
 * PostgreSQL이나 진짜 모델이 없어도 돌아가는 것이 목적이다 (진짜 연동은 수동 E2E로 따로 확인).
 */
@SpringBootTest
@AutoConfigureMockMvc
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:h2:mem:curtailment_test;DB_CLOSE_DELAY=-1",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "spring.jpa.show-sql=false"
})
class PredictionApiIntegrationTest {

    private static final String TRAINED_BEFORE = "2026-09-27T06:22:32+00:00";
    private static final String TRAINED_AFTER = "2026-10-10T09:00:00+00:00";

    private static HttpServer aiServer;
    private static volatile int aiStatus;        // 200(정상) / 422 / 500
    private static volatile String trainedAt;    // 가짜 AI 서버가 돌려줄 model_trained_at

    @Autowired
    MockMvc mvc;

    @DynamicPropertySource
    static void aiServerProperties(DynamicPropertyRegistry registry) throws IOException {
        aiServer = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        aiServer.createContext("/predict", exchange -> {
            exchange.getRequestBody().readAllBytes();
            int status;
            String type;
            String body;
            switch (aiStatus) {
                case 422 -> {
                    status = 422;
                    type = "application/json";
                    body = "{\"error_code\":\"INVALID_HOUR_SET\",\"message\":\"weather must contain hours 1-24\"}";
                }
                case 500 -> {
                    status = 500;
                    type = "text/plain";
                    body = "Internal Server Error"; // 실제 AI 서버 500은 JSON이 아니라 평문이다
                }
                default -> {
                    status = 200;
                    type = "application/json";
                    body = okBody();
                }
            }
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", type + "; charset=utf-8");
            exchange.sendResponseHeaders(status, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        });
        aiServer.start();
        registry.add("ai-server.base-url", () -> "http://127.0.0.1:" + aiServer.getAddress().getPort());
    }

    @AfterAll
    static void stopAiServer() {
        aiServer.stop(0);
    }

    @BeforeEach
    void resetAiServer() {
        aiStatus = 200;
        trainedAt = TRAINED_BEFORE;
    }

    @Test
    @DisplayName("POST는 24건을 저장하고, 24시는 다음날 00:00으로 저장한다")
    void post_saves24Rows_and_hour24BecomesNextDayMidnight() throws Exception {
        String name = "t1-" + UUID.randomUUID();
        long regionId = registerRegion(name);

        predict(name).andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(24));

        String body = mvc.perform(get("/api/predictions/" + regionId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(24))
                .andExpect(jsonPath("$[0].targetHour").value("2026-10-11T00:00:00"))   // 24시
                .andExpect(jsonPath("$[23].targetHour").value("2026-10-10T01:00:00"))  // 1시
                .andExpect(jsonPath("$[0].regionName").value(name))
                .andExpect(jsonPath("$[0].modelUsed").value("classifier_wind_demand_crossp_lr_calibrated_sigmoid"))
                .andExpect(jsonPath("$[0].modelTrainedAt").value(TRAINED_BEFORE))
                .andExpect(jsonPath("$[0].operationalThreshold").value(0.01))
                .andExpect(jsonPath("$[0].operationalThresholdReliable").value(false))
                .andReturn().getResponse().getContentAsString(StandardCharsets.UTF_8);

        // 풍력 응답은 255자를 넘는 note가 온다 - TEXT 컬럼이 아니면 여기서 저장이 실패한다
        String note = JsonPath.read(body, "$[0].note");
        assertThat(note.length()).isGreaterThan(255);
    }

    @Test
    @DisplayName("GET은 시각별 최신 1건만 주고, history=true는 쌓인 전체 이력을 준다")
    void get_returnsLatestPerHour_and_historyKeepsEverything() throws Exception {
        String name = "t2-" + UUID.randomUUID();
        long regionId = registerRegion(name);

        trainedAt = TRAINED_BEFORE;
        predict(name).andExpect(status().isOk());
        trainedAt = TRAINED_AFTER; // 재학습 후 같은 날짜를 다시 예측
        predict(name).andExpect(status().isOk());

        mvc.perform(get("/api/predictions/" + regionId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(24))
                .andExpect(jsonPath("$[*].modelTrainedAt", everyItem(is(TRAINED_AFTER))));

        mvc.perform(get("/api/predictions/" + regionId + "?history=true"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(48));
    }

    @Test
    @DisplayName("AI 서버의 422(error_code JSON)는 그대로 전달하고, 아무것도 저장하지 않는다")
    void aiServer422_isRelayedAsIs_and_nothingIsSaved() throws Exception {
        String name = "t3-" + UUID.randomUUID();
        long regionId = registerRegion(name);

        aiStatus = 422;
        predict(name).andExpect(status().isUnprocessableEntity())
                .andExpect(content().string(containsString("INVALID_HOUR_SET")));

        mvc.perform(get("/api/predictions/" + regionId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(0));
    }

    @Test
    @DisplayName("AI 서버의 500(평문)은 JSON ErrorResponse로 감싸서 내려준다")
    void aiServer500_isWrappedAsJsonError() throws Exception {
        String name = "t4-" + UUID.randomUUID();
        registerRegion(name);

        aiStatus = 500;
        predict(name).andExpect(status().isInternalServerError())
                .andExpect(content().string(containsString("AI_SERVER_ERROR")));
    }

    @Test
    @DisplayName("등록되지 않은 지역은 AI 서버를 부르기 전에 400으로 끝난다")
    void unknownRegion_returns400() throws Exception {
        predict("no-such-region-" + UUID.randomUUID())
                .andExpect(status().isBadRequest())
                .andExpect(content().string(containsString("INVALID_REQUEST")));
    }

    // ----------------------------------------------------------------- 헬퍼

    private long registerRegion(String name) throws Exception {
        String body = mvc.perform(post("/api/regions").contentType(APPLICATION_JSON)
                        .content("{\"name\":\"" + name + "\",\"energySource\":\"WIND\"}"))
                .andExpect(status().is2xxSuccessful())
                .andReturn().getResponse().getContentAsString(StandardCharsets.UTF_8);
        return ((Number) JsonPath.read(body, "$.id")).longValue();
    }

    private ResultActions predict(String regionName) throws Exception {
        return mvc.perform(post("/api/predictions").contentType(APPLICATION_JSON)
                .content("{\"energy_type\":\"wind\",\"region\":\"" + regionName
                        + "\",\"target_date\":\"2026-10-10\"}"));
    }

    /** 진짜 AI 서버 POST /predict 응답과 같은 모양(snake_case) - hour 1~24, 긴 note 포함. */
    private static String okBody() {
        StringBuilder hourly = new StringBuilder();
        for (int h = 1; h <= 24; h++) {
            if (h > 1) {
                hourly.append(',');
            }
            hourly.append("{\"hour\":").append(h)
                    .append(",\"generation_forecast_mwh\":").append(10.0 + h)
                    .append(",\"curtailment_probability\":").append(0.01 * h)
                    .append(",\"expected_curtailment_mwh\":").append(0.5 * h)
                    .append('}');
        }
        String note = "긴 note ".repeat(60); // 420자 - 255자 제한 컬럼이면 저장 실패
        return "{\"energy_type\":\"wind\",\"region\":\"x\",\"target_date\":\"2026-10-10\","
                + "\"hourly\":[" + hourly + "],"
                + "\"model_used\":\"classifier_wind_demand_crossp_lr_calibrated_sigmoid\","
                + "\"model_trained_at\":\"" + trainedAt + "\","
                + "\"operational_threshold\":0.01,\"operational_threshold_reliable\":false,"
                + "\"note\":\"" + note + "\"}";
    }
}
