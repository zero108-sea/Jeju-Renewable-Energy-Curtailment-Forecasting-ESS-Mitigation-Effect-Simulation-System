#!/bin/bash
# KIM 예보 일일 수집을 launchd에 등록한다.
#
# [키를 어떻게 다루는가]
# plist에 키를 넣지 않는다. 스크립트가 실행 시점에 키체인에서 직접 읽는다
# (app/services/credentials.py). 그래서 이 파일에도, plist에도, 로그에도 키가 남지 않는다.
#
# [왜 매일 돌려야 하나]
# API가 과거 예보를 보관하지 않는다 — 응답이 오는 가장 오래된 분석시각이 하루 전이다.
# 놓친 날은 되돌릴 수 없고, 그만큼 검증 표본이 영구히 비게 된다.
#
# [왜 10시부터 23시까지 매시인가 — 네 번으로도 놓쳤다]
# 처음에는 10시 한 번만 걸었는데 첫날 실행되지 않았다(launchd runs=0). StartCalendarInterval은
# 그 시각에 맥이 깨어 있어야 발동하고, 잠들어 있으면 그 실행을 건너뛴다(깬 뒤 보충 실행도
# 없었다). 그래서 10/12/14/16시 네 번으로 늘렸다.
#
# **그런데 그것으로도 놓쳤다.** 10-04·10-05 이틀 모두 낮 시간대에 덮개를 닫아두어 네 슬롯이
# 전부 수면 중에 지나갔고, 거래일 10-06·10-07분 예보가 영구히 비었다(10-07분은 10-06에
# 수동 실행으로 겨우 건졌다). 노트북에서 '특정 시각'을 믿을 수 없다는 뜻이다.
#
# 그래서 안전창(아래 참고) 전체에 매시로 깔았다 — '10시 이후 처음 깨어 있는 시각에 받는다'에
# 가깝게 동작한다. 이미 24시간을 받아둔 (거래일, 분석시각)이면 스크립트가 API를 호출하지 않고
# 끝내므로 열네 번 걸어도 네트워크 비용은 첫 성공 한 번뿐이다.
#
# [왜 10시부터인가]
# 거래일 D의 자료를 받으려면 D-1 09:00 KST(단기예보 08시 발표 + KIM 생산 완료) 이후여야 하고,
# KIM 보관이 만료되는 D 07:00 KST 전이어야 한다. 10시는 그 창의 앞쪽이고 하루전시장
# 입찰마감(D-1 11시)과도 같은 타이밍이다.
#
# 사용:  bash scripts/setup_daily_collect.sh          등록
#        bash scripts/setup_daily_collect.sh remove   해제
set -euo pipefail

LABEL="com.jeju.kim-collect"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
# [왜 로그가 저장소 밖인가 — 2026-10-10]
# 처음에는 $REPO/logs/에 뒀는데 **24번 연속 exit 78(EX_CONFIG)로 실패했다.** 로그 파일에
# 한 글자도 안 남아 원인이 안 보였는데, 그게 단서였다 — 프로세스가 시작조차 못 한 것이다.
# 이 저장소는 ~/Downloads 아래에 있고 그곳은 macOS가 TCC로 보호한다. launchd가 띄우는
# 프로세스에는 그 접근권이 없어서 StandardOutPath를 **열지 못하고** 그대로 죽는다.
# 같은 파이썬·같은 WorkingDirectory로 로그 경로만 ~/Library/Logs로 바꿔 시험하니 exit 0이었다.
# (작업디렉터리가 ~/Downloads인 것은 괜찮다 — 막히는 것은 launchd가 직접 여는 로그 파일이다.)
LOG="$HOME/Library/Logs/kim_collect.log"

if [ "${1:-}" = "remove" ]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "해제했습니다: $LABEL"
    exit 0
fi

if ! security find-generic-password -s KMA_AUTH_KEY -w >/dev/null 2>&1; then
    echo "❌ 키체인에 KMA_AUTH_KEY가 없습니다. 먼저 한 번만 등록하세요:"
    echo
    echo "   security add-generic-password -a \"\$USER\" -s KMA_AUTH_KEY \\"
    echo "     -T /usr/bin/security -U -w"
    echo
    echo "   실행하면 비밀번호를 물어봅니다. 거기에 발급받은 키를 붙여넣으세요."
    echo "   (-w 뒤에 키를 직접 쓰면 셸 히스토리에 남습니다)"
    exit 1
fi
echo "✅ 키체인에서 KMA_AUTH_KEY 확인"

mkdir -p "$(dirname "$LOG")" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PY</string>
    <string>$REPO/scripts/kim_validation_log.py</string>
    <string>collect</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>10</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>12</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>13</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>14</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>15</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>16</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>17</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>19</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>23</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "✅ 등록 완료 — 매일 10~23시 매시 실행 (이미 받은 날은 API 호출 없이 끝남)"
echo "   로그:   $LOG"
echo "   상태:   launchctl print gui/$(id -u)/$LABEL | head -20"
echo "   즉시실행: launchctl kickstart -k gui/$(id -u)/$LABEL"
echo "   해제:   bash scripts/setup_daily_collect.sh remove"
echo
echo "⚠ 10~23시 내내 맥이 꺼져 있거나 잠들어 있으면 그 날은 건너뜁니다."
echo "  API 보관이 하루뿐이라 놓친 날은 되돌릴 수 없습니다. 며칠에 한 번 로그를 확인하고,"
echo "  놓친 날이 보이면 그날 07시 전까지는 collect로 복구할 수 있습니다."
