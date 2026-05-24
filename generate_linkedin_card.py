# generate_linkedin_card.py (v2)
# 변경:
#   1. 전체 텍스트 크기 대폭 확대 (모바일 가독성)
#   2. 뉴스 제목 + 핵심 내용(body) 함께 표시
#   3. "기사 보기" 링크 제거
#   4. 뉴스 3개로 줄이고 내용 충실히

import os
import json
import re
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
KST = timezone(timedelta(hours=9))


# ── 유틸 함수 ──────────────────────────────────────────────

def get_week_label() -> str:
    now = datetime.now(KST)
    week_num = (now.day - 1) // 7 + 1
    return f"{now.year}년 {now.month}월 {week_num}주차"


def fetch_reports(supabase: Client, week_label: str) -> dict:
    resp = supabase.table('weekly_reports') \
        .select('*') \
        .eq('week_label', week_label) \
        .execute()
    result = {}
    for row in (resp.data or []):
        result[row['category']] = {
            'report_text':   row.get('report_text', ''),
            'sections_json': json.loads(row.get('sections_json') or '{}'),
        }
    return result


def parse_summary(report_text: str) -> list[str]:
    summary, in_summary = [], False
    for line in report_text.split('\n'):
        line = line.strip()
        if '핵심 요약' in line:
            in_summary = True; continue
        if in_summary and line.startswith('##'): break
        if in_summary and line.startswith('-'):
            text = re.sub(r'^[-•]\s*', '', line)
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text).strip()
            if text: summary.append(text)
    return summary[:3]


def parse_actions(report_text: str) -> list[str]:
    actions, in_action = [], False
    for line in report_text.split('\n'):
        line = line.strip()
        if '실무 적용' in line:
            in_action = True; continue
        if in_action and line.startswith('##'): break
        if in_action and line and (
            line[0].isdigit() or line.startswith('-') or line.startswith('•')
        ):
            text = re.sub(r'^[\d\-•.]\s*', '', line)
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            text = re.sub(r'\s*[\(\[]?관련:?\s*https?://\S+[\)\]]?', '', text)
            text = re.sub(r'\s*\(?출처:.*', '', text)
            text = re.sub(r'\s*참고 URL:.*', '', text)
            text = re.sub(r'\s*https?://\S+', '', text)
            text = text.strip().lstrip('.').strip()
            text = re.sub(r'\([^)]*$', '', text).strip()
            if text and len(text) > 10:
                actions.append(text)
    return actions[:3]


# 노이즈 필터
NOISE_KEYWORDS = [
    'battery skills', 'nfl', 'motoring', 'innovate uk',
    'apprenticeship', 'healthcare', 'pharmacy',
    'student fellows', 'martha stewart', 'aixalpha', 'xrp',
    '주가', 'valuation', '밸류에이션',
]

def parse_news_with_body(report_text: str, max_items: int = 3) -> list[dict]:
    """뉴스 제목 + 핵심 설명 1줄 추출 (섹션별 균등)"""
    section_items = {}
    current_icon  = '📌'
    current_title = None
    current_body_lines = []
    seen_titles   = set()

    def flush_item():
        nonlocal current_title, current_body_lines
        if not current_title:
            return
        title_lower = current_title.lower()
        if any(kw in title_lower for kw in NOISE_KEYWORDS):
            current_title = None; current_body_lines = []; return

        title_key = re.sub(r'[^\w가-힣]', '', title_lower)[:20]
        if title_key in seen_titles:
            current_title = None; current_body_lines = []; return

        # body: 들여쓰기된 설명 첫 줄 사용
        body = ''
        for bl in current_body_lines:
            b = re.sub(r'\*\*(.*?)\*\*', r'\1', bl)
            b = re.sub(r'\s*출처:.*', '', b)
            b = re.sub(r'\s*https?://\S+', '', b)
            b = b.strip()
            if b and len(b) > 10:
                body = b[:80]
                break

        if len(current_title) > 10:
            seen_titles.add(title_key)
            if current_icon not in section_items:
                section_items[current_icon] = []
            if len(section_items[current_icon]) < 2:
                section_items[current_icon].append({
                    'icon':  current_icon,
                    'title': current_title,
                    'body':  body,
                })
        current_title = None
        current_body_lines = []

    for line in report_text.split('\n'):
        raw  = line
        line = line.strip()

        # 섹션 감지
        for emoji in ['🚀', '🔄', '💰', '🛠', '📊']:
            if line.startswith(f'### {emoji}'):
                flush_item()
                current_icon = emoji
                if current_icon not in section_items:
                    section_items[current_icon] = []
                break

        # 제목 감지 (- ** 로 시작)
        if line.startswith('- **') and '**' in line[4:]:
            flush_item()
            m = re.search(r'\*\*(.*?)\*\*', line)
            if m:
                current_title = m.group(1)
                current_body_lines = []

        # 본문 수집 (들여쓰기 있는 줄, - 로 시작하되 제목 아님)
        elif current_title and (raw.startswith('  ') or raw.startswith('\t')):
            stripped = line.lstrip('-•').strip()
            if stripped and not stripped.startswith('출처') and 'https://' not in stripped:
                current_body_lines.append(stripped)

    flush_item()

    # 섹션별 균등 추출
    icons_order = ['🚀', '🔄', '💰', '🛠', '📊']
    result = []
    for icon in icons_order:
        items = section_items.get(icon, [])
        if items: result.append(items[0])
        if len(result) >= max_items: break
    # 부족하면 2번째 항목으로 채움
    if len(result) < max_items:
        for icon in icons_order:
            items = section_items.get(icon, [])
            if len(items) >= 2:
                cand = items[1]
                if cand not in result:
                    result.append(cand)
            if len(result) >= max_items: break

    return result[:max_items]


def clean_insight(text: str) -> str:
    text = re.sub(r'\s*[\(\[]?관련:?\s*https?://\S+[\)\]]?', '', text)
    text = re.sub(r'\s*출처:.*', '', text)
    text = re.sub(r'\s*참고 URL:.*', '', text)
    text = re.sub(r'\s*https?://\S+', '', text)
    text = text.strip().lstrip('.').strip()
    text = re.sub(r'\([^)]*$', '', text).strip()
    # 글자 수 제한 없이 전체 반환 (화면에서 JS로 크기 조절)
    return text


# ── 슬라이드 생성 ──────────────────────────────────────────

def build_category_slide(cat, color, headline, slide_num, data):
    report_text = data.get('report_text', '')

    summary      = parse_summary(report_text)
    summary_text = summary[0] if summary else ''

    top_news = parse_news_with_body(report_text, max_items=3)

    actions      = parse_actions(report_text)
    insight_raw  = actions[0] if actions else ''
    insight      = clean_insight(insight_raw)

    color_map = {
        'ai':  ('var(--ai)',  'rgba(123,110,246,0.12)', 'linear-gradient(90deg,var(--ai),#a78bfa)'),
        'de':  ('var(--de)',  'rgba(0,201,167,0.12)',   'linear-gradient(90deg,var(--de),#34d8b5)'),
        'rpa': ('var(--rpa)', 'rgba(255,107,107,0.12)', 'linear-gradient(90deg,var(--rpa),#ff9a9a)'),
    }
    c_main, c_bg, c_grad = color_map.get(color, color_map['ai'])

    cat_label_map = {
        'AI': 'Artificial Intelligence',
        '데이터엔지니어링': 'Data Engineering',
        'RPA': 'RPA & Automation',
    }
    cat_label = cat_label_map.get(cat, cat)

    # 뉴스 항목 HTML (제목 + body, 링크 없음)
    news_rows_html = ''
    for news in top_news:
        body_html = f'<div class="news-body">{news["body"]}</div>' if news.get('body') else ''
        news_rows_html += f'''
      <div class="news-row">
        <span class="news-icon">{news["icon"]}</span>
        <div class="news-content">
          <div class="news-title">{news["title"]}</div>
          {body_html}
        </div>
      </div>'''

    insight_id   = f'insight-{slide_num}'
    insight_html = f'''
    <div class="insight-box">
      <div class="insight-label">💡 THIS WEEK'S INSIGHT</div>
      <div class="insight-text" id="{insight_id}">{insight}</div>
    </div>''' if insight else ''

    now = datetime.now(KST)
    week_num = (now.day - 1) // 7 + 1
    footer_week = f"{now.year}.{now.month:02d} W{week_num}"

    return f'''
<div class="slide-label">SLIDE 0{slide_num} / 05 — {cat}</div>
<div class="slide" style="background:var(--card-bg);">
  <div style="position:absolute;top:0;left:0;right:0;height:6px;background:{c_grad};"></div>
  <div style="position:absolute;width:500px;height:500px;border-radius:50%;
    background:radial-gradient(circle,{c_bg} 0%,transparent 70%);top:-100px;right:-100px;"></div>

  <div style="position:relative;z-index:2;height:100%;padding:48px 72px 36px;
    display:flex;flex-direction:column;">

    <!-- 헤더 -->
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;flex-shrink:0;">
      <div style="width:16px;height:16px;border-radius:50%;background:{c_main};flex-shrink:0;"></div>
      <span style="font-family:'JetBrains Mono',monospace;font-size:18px;letter-spacing:4px;
        text-transform:uppercase;color:{c_main};">{cat_label}</span>
      <span style="margin-left:auto;font-family:'Bebas Neue',sans-serif;font-size:24px;
        letter-spacing:2px;opacity:0.25;color:#fff;">0{slide_num} / 05</span>
    </div>

    <!-- 카테고리 타이틀 -->
    <div style="font-family:'Bebas Neue',sans-serif;font-size:76px;line-height:1;
      letter-spacing:3px;color:#fff;margin-bottom:20px;flex-shrink:0;">{headline}<br>WEEKLY</div>

    <!-- 핵심 요약 -->
    <div style="background:rgba(255,255,255,0.05);border-left:4px solid {c_main};
      padding:18px 22px;margin-bottom:20px;border-radius:0 4px 4px 0;flex-shrink:0;">
      <div style="font-size:21px;line-height:1.5;color:var(--text);font-weight:400;">{summary_text}</div>
    </div>

    <!-- 뉴스 목록 -->
    <div style="flex:1;display:flex;flex-direction:column;gap:0;min-height:0;overflow:hidden;">
      {news_rows_html}
    </div>

    <!-- 인사이트 (flex 내부, 푸터 위에 자연스럽게 위치) -->
    {insight_html}

    <!-- 푸터 -->
    <div style="display:flex;justify-content:space-between;align-items:center;
      margin-top:16px;flex-shrink:0;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:15px;
        letter-spacing:2px;color:var(--muted);">IT WEEKLY · {footer_week}</span>
      <span style="font-family:'Bebas Neue',sans-serif;font-size:22px;
        letter-spacing:3px;color:rgba(255,255,255,0.12);">0{slide_num} / 05</span>
    </div>

  </div>
</div>'''


def generate_card_html(reports: dict, week_label: str) -> str:
    now      = datetime.now(KST)
    week_num = (now.day - 1) // 7 + 1
    date_str = f"{now.year}.{now.month:02d}.{now.day:02d}"
    week_str = f"{now.year} MAY WEEK {week_num}"

    categories = [
        ('AI',            'ai',  'AI',   2),
        ('데이터엔지니어링', 'de',  'DATA', 3),
        ('RPA',           'rpa', 'RPA',  4),
    ]

    category_slides = ''
    for cat, color, headline, slide_num in categories:
        data = reports.get(cat, {'report_text': '', 'sections_json': {}})
        category_slides += build_category_slide(cat, color, headline, slide_num, data)

    # 마무리 슬라이드 액션 포인트
    all_actions = []
    for cat, _, _, _ in categories:
        data    = reports.get(cat, {})
        actions = parse_actions(data.get('report_text', ''))
        if actions:
            all_actions.append({'cat': cat, 'text': clean_insight(actions[0])})

    actions_html = ''
    for i, action in enumerate(all_actions[:3], 1):
        actions_html += f'''
      <li style="display:flex;gap:20px;padding:22px 24px;background:rgba(255,255,255,0.05);
        border-radius:4px;align-items:flex-start;">
        <span style="font-family:'Bebas Neue',sans-serif;font-size:40px;line-height:1;
          color:var(--accent);flex-shrink:0;min-width:36px;">0{i}</span>
        <div style="padding-top:4px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:14px;
            color:var(--accent);letter-spacing:2px;margin-bottom:6px;">[{action["cat"]}]</div>
          <div style="font-size:20px;color:var(--text);line-height:1.5;">{action["text"]}</div>
        </div>
      </li>'''

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>IT 주간 동향 카드뉴스 | {week_label}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Bebas+Neue&family=JetBrains+Mono:wght@400;600&display=swap');
  :root {{
    --ai:#7B6EF6; --de:#00C9A7; --rpa:#FF6B6B;
    --dark:#0D0D14; --card-bg:#111118;
    --text:#F0F0F8; --muted:#8888AA; --accent:#F5C842;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Noto Sans KR',sans-serif; background:#1a1a2e;
    display:flex; flex-direction:column; align-items:center; padding:40px 20px; gap:32px; }}
  .slide {{ width:1080px; height:1080px; position:relative; overflow:hidden;
    flex-shrink:0; page-break-after:always; }}
  .slide-label {{ color:#555577; font-family:'JetBrains Mono',monospace; font-size:13px;
    text-align:center; margin-bottom:8px; letter-spacing:2px; }}
  @media print {{
    body {{ background:white; padding:0; gap:0; }}
    .slide-label {{ display:none; }}
    .slide {{ page-break-after:always; }}
  }}

  /* 뉴스 행 */
  .news-row {{
    display:flex; gap:20px; padding:18px 0;
    border-bottom:1px solid rgba(255,255,255,0.07);
    align-items:flex-start;
  }}
  .news-row:last-child {{ border-bottom:none; }}
  .news-icon {{ font-size:26px; flex-shrink:0; margin-top:2px; }}
  .news-content {{ flex:1; }}
  .news-title {{
    font-size:22px; font-weight:600; color:#fff;
    line-height:1.35; margin-bottom:6px;
    word-break:keep-all; overflow-wrap:break-word;
  }}
  .news-body {{
    font-size:18px; color:var(--muted); line-height:1.5;
    word-break:keep-all; overflow-wrap:break-word;
  }}

  /* 인사이트 박스 */
  .insight-box {{
    margin-top:16px;
    margin-bottom:12px;
    padding:22px 26px;
    background:rgba(245,200,66,0.08);
    border:1px solid rgba(245,200,66,0.25);
    border-radius:4px;
    /* 높이 고정 없음 - 내용에 맞게 자동 조절 */
  }}
  .insight-label {{
    font-family:'JetBrains Mono',monospace; font-size:13px;
    letter-spacing:3px; color:var(--accent); margin-bottom:8px;
  }}
  .insight-text {{
    font-size:20px; color:var(--text); line-height:1.5;
  }}
</style>
</head>
<body>

<!-- SLIDE 1 / 표지 -->
<div class="slide-label">SLIDE 01 / 05 — 표지</div>
<div class="slide" style="background:var(--dark);">
  <div style="position:absolute;inset:0;
    background-image:linear-gradient(rgba(123,110,246,0.08) 1px,transparent 1px),
    linear-gradient(90deg,rgba(123,110,246,0.08) 1px,transparent 1px);
    background-size:60px 60px;"></div>
  <div style="position:absolute;width:700px;height:700px;border-radius:50%;
    background:radial-gradient(circle,rgba(123,110,246,0.18) 0%,transparent 65%);
    top:-150px;left:-150px;"></div>
  <div style="position:absolute;width:500px;height:500px;border-radius:50%;
    background:radial-gradient(circle,rgba(0,201,167,0.12) 0%,transparent 65%);
    bottom:-100px;right:-100px;"></div>

  <div style="position:relative;z-index:2;height:100%;display:flex;flex-direction:column;
    justify-content:space-between;padding:80px 88px;">
    <div>
      <div style="display:inline-flex;align-items:center;gap:10px;
        background:rgba(245,200,66,0.12);border:1px solid rgba(245,200,66,0.3);
        color:var(--accent);font-family:'JetBrains Mono',monospace;
        font-size:18px;letter-spacing:3px;padding:12px 24px;">
        WEEKLY TREND REPORT
      </div>
    </div>
    <div>
      <div style="font-family:'Bebas Neue',sans-serif;font-size:130px;
        line-height:0.92;letter-spacing:6px;color:#fff;">IT<br>
        <span style="background:linear-gradient(135deg,var(--ai),var(--de));
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;">TREND</span><br>DIGEST
      </div>
      <div style="font-size:26px;color:var(--muted);font-weight:300;
        letter-spacing:1px;margin-top:20px;line-height:1.6;">
        매주 놓치면 안 될
        <strong style="color:var(--text);font-weight:500;">AI · 데이터엔지니어링 · RPA</strong><br>
        핵심 동향을 한 장으로 정리합니다.
      </div>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:flex-end;">
      <div style="display:flex;gap:14px;">
        <div style="padding:10px 22px;border-radius:2px;font-size:17px;font-weight:700;
          letter-spacing:1px;background:rgba(123,110,246,0.2);color:var(--ai);
          border:1px solid rgba(123,110,246,0.4);">AI</div>
        <div style="padding:10px 22px;border-radius:2px;font-size:17px;font-weight:700;
          letter-spacing:1px;background:rgba(0,201,167,0.2);color:var(--de);
          border:1px solid rgba(0,201,167,0.4);">DATA ENG</div>
        <div style="padding:10px 22px;border-radius:2px;font-size:17px;font-weight:700;
          letter-spacing:1px;background:rgba(255,107,107,0.2);color:var(--rpa);
          border:1px solid rgba(255,107,107,0.4);">RPA</div>
      </div>
      <div style="text-align:right;font-family:'JetBrains Mono',monospace;
        font-size:17px;color:var(--muted);letter-spacing:1px;line-height:1.8;">
        {date_str}<br>{week_str}
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 2, 3, 4 -->
{category_slides}

<!-- SLIDE 5 / 마무리 -->
<div class="slide-label">SLIDE 05 / 05 — 마무리</div>
<div class="slide" style="background:var(--dark);">
  <div style="position:absolute;inset:0;
    background-image:linear-gradient(rgba(255,255,255,0.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,0.03) 1px,transparent 1px);
    background-size:54px 54px;"></div>
  <div style="position:absolute;bottom:0;left:0;right:0;height:7px;
    background:linear-gradient(90deg,var(--ai) 0%,var(--de) 50%,var(--rpa) 100%);"></div>

  <div style="position:relative;z-index:2;height:100%;padding:80px 88px;
    display:flex;flex-direction:column;justify-content:space-between;">
    <div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
        letter-spacing:4px;color:var(--accent);margin-bottom:24px;">
        THIS WEEK'S ACTION ITEMS</div>
      <div style="font-family:'Bebas Neue',sans-serif;font-size:96px;
        line-height:0.92;letter-spacing:4px;color:#fff;margin-bottom:40px;">
        지금 바로<br>챙겨야 할<br>3가지</div>
      <ul style="list-style:none;display:flex;flex-direction:column;gap:16px;">
        {actions_html}
      </ul>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:flex-end;">
      <div style="font-family:'Bebas Neue',sans-serif;font-size:32px;
        letter-spacing:4px;color:rgba(255,255,255,0.18);">TREND DIGEST</div>
      <div style="text-align:right;">
        <div style="font-size:18px;color:var(--muted);line-height:1.7;">
          매주 월요일 업데이트<br>
          <strong style="color:var(--accent);font-size:20px;">팔로우하고 놓치지 마세요 →</strong>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
window.addEventListener('load', function() {{
  // 인사이트 텍스트가 박스를 넘치면 폰트 크기를 줄임
  ['insight-2', 'insight-3', 'insight-4'].forEach(function(id) {{
    var el = document.getElementById(id);
    if (!el) return;
    var box = el.closest('.insight-box');
    if (!box) return;
    var size = 20;
    el.style.fontSize = size + 'px';
    // 텍스트가 박스 높이를 넘으면 줄임
    while (el.scrollHeight > el.clientHeight + 2 && size > 13) {{
      size -= 0.5;
      el.style.fontSize = size + 'px';
    }}
  }});
}});
</script>
</body>
</html>'''


def main():
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    if not all([supabase_url, supabase_key]):
        raise ValueError("환경변수 누락: SUPABASE_URL, SUPABASE_KEY 확인")

    supabase   = create_client(supabase_url, supabase_key)
    week_label = get_week_label()
    print(f"카드뉴스 생성: {week_label}")

    reports = fetch_reports(supabase, week_label)
    if not reports:
        print(f"❌ {week_label} 리포트가 DB에 없습니다.")
        return

    print(f"✅ {list(reports.keys())} 카테고리 로드 완료")

    html     = generate_card_html(reports, week_label)
    filename = (
        f"linkedin_card_{week_label}"
        .replace(' ', '_').replace('년', '').replace('월', '').replace('주차', 'w')
        + '.html'
    )
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ 저장 완료: {output}")
    print(f"\n📌 PDF 변환:")
    print(f"   Chrome에서 {filename} 열기 → Cmd+P → 여백:없음 + 배경그래픽 체크 → PDF 저장")
    print(f"   링크드인 게시물 → 문서 업로드 → PDF 선택")


if __name__ == "__main__":
    main()