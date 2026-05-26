#!/bin/bash
# run_weekly.sh - 매주 월요일 실행 (리포트 생성)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
OUTPUT_DIR="$PROJECT_DIR/output"
DATE=$(date '+%Y-%m-%d')
WEEK=$(date '+%Y_W%V')

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "  TREND-BOX 주간 리포트 생성 시작"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"


# ── 1. 뉴스 수집 ─────────────────────────────
echo ""
echo "[1/2] 뉴스 수집 시작..."
uv run "$PROJECT_DIR/rss-scraper.py" >> "$LOG_DIR/scraper_$DATE.log" 2>&1
if [ $? -eq 0 ]; then
    echo "      ✅ 뉴스 수집 완료"
else
    echo "      ❌ 뉴스 수집 실패 (로그: logs/scraper_$DATE.log)"
fi

# ── 2. 논문 수집 ─────────────────────────────
echo ""
echo "[2/2] arXiv 논문 수집 시작..."
uv run "$PROJECT_DIR/arxiv_scraper.py" >> "$LOG_DIR/arxiv_$DATE.log" 2>&1
if [ $? -eq 0 ]; then
    echo "      ✅ 논문 수집 완료"
else
    echo "      ❌ 논문 수집 실패 (로그: logs/arxiv_$DATE.log)"
fi

# ── 3. 뉴스 + 논문 처리 → DB 저장 ───────────
echo ""
echo "[1/3] 뉴스+논문 처리 및 리포트 생성..."
uv run "$PROJECT_DIR/news_data_processor.py" >> "$LOG_DIR/processor_$DATE.log" 2>&1
if [ $? -eq 0 ]; then
    echo "      ✅ 리포트 DB 저장 완료"
else
    echo "      ❌ 처리 실패 (로그: logs/processor_$DATE.log)"
    exit 1
fi

# ── 4. 리포트 HTML 생성 ──────────────────────
echo ""
echo "[2/3] 리포트 HTML 생성..."
uv run "$PROJECT_DIR/generate_report_html.py" >> "$LOG_DIR/report_html_$DATE.log" 2>&1
if [ $? -eq 0 ]; then
    echo "      ✅ 리포트 HTML 생성 완료"
    # output 폴더로 이동
    mv "$PROJECT_DIR"/weekly_report_*.html "$OUTPUT_DIR/" 2>/dev/null || true
else
    echo "      ❌ 리포트 HTML 생성 실패"
fi

# ── 5. 카드뉴스 HTML 생성 ────────────────────
echo ""
echo "[3/3] 링크드인 카드뉴스 생성..."
uv run "$PROJECT_DIR/generate_linkedin_card.py" >> "$LOG_DIR/card_$DATE.log" 2>&1
if [ $? -eq 0 ]; then
    echo "      ✅ 카드뉴스 HTML 생성 완료"
    mv "$PROJECT_DIR"/linkedin_card_*.html "$OUTPUT_DIR/" 2>/dev/null || true
else
    echo "      ❌ 카드뉴스 생성 실패"
fi

echo ""
echo "======================================"
echo "  주간 리포트 생성 완료: $(date '+%H:%M:%S')"
echo ""
echo "  📁 생성된 파일:"
ls -lh "$OUTPUT_DIR"/*.html 2>/dev/null | awk '{print "     " $NF}'
echo ""
echo "  📌 다음 단계:"
echo "     1. output/ 폴더에서 HTML 파일 확인"
echo "     2. Chrome에서 카드뉴스 열고 Cmd+P → PDF 저장"
echo "     3. 링크드인 게시물에 PDF 업로드"
echo "======================================"ㅕ