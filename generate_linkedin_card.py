# generate_linkedin_card_v4.py
# 수정사항:
#   - 논문 슬라이드: 3열 가로 그리드로 레이아웃 균형 개선
#   - URL 연결: 뉴스/논문 제목에 href 링크 추가
#   - Insight: why/apply 구분 제거, 자연스러운 1문장으로 통일

import os
import json
import re
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
KST           = timezone(timedelta(hours=9))
MODEL_FAST    = "gpt-5-nano"
MODEL_QUALITY = "gpt-5.4-nano"


# ── 유틸 ────────────────────────────────────────────────────

def get_week_label() -> str:
    now      = datetime.now(KST)
    week_num = (now.day - 1) // 7 + 1
    return f"{now.year}년 {now.month}월 {week_num}주차"


def truncate(text: str, max_len: int = 80) -> str:
    if not text:
        return ""
    text = str(text).strip()
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


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


def fetch_papers(supabase: Client, days: int = 7) -> dict:
    now   = datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).date().isoformat()
    result = {}
    for cat in ['AI', '데이터엔지니어링', 'RPA']:
        try:
            resp = supabase.table('arxiv_papers') \
                .select('title, abstract, url, published_date, category') \
                .eq('category', cat) \
                .gte('published_date', start) \
                .order('published_date', desc=True) \
                .limit(10) \
                .execute()
            result[cat] = resp.data or []
        except Exception as e:
            print(f"  논문 조회 실패 [{cat}]: {e}")
            result[cat] = []
    return result


def parse_summary(report_text: str) -> list:
    summary, in_summary = [], False
    for line in report_text.split('\n'):
        line = line.strip()
        if '핵심 요약' in line:
            in_summary = True; continue
        if in_summary and line.startswith('##'): break
        if in_summary and line.startswith('-'):
            text = re.sub(r'^[-•]\s*', '', line)
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text).strip()
            text = re.sub(r'\s*https?://\S+', '', text).strip()
            if text: summary.append(text)
    return summary[:3]


# ── sections_json에서 뉴스 추출 ──────────────────────────────

def extract_news_from_sections(
    sections_json: dict, max_items: int = 5
) -> list:
    priority = [
        'domestic', 'global', 'new_services', 'updates',
        'trends', 'infrastructure', 'investment',
    ]
    icon_map = {
        'domestic':       '📰',
        'global':         '🌐',
        'new_services':   '🚀',
        'updates':        '🔄',
        'investment':     '💰',
        'infrastructure': '🛠',
        'trends':         '📊',
    }
    seen_titles = set()
    result      = []

    for section_key in priority:
        items = sections_json.get(section_key, [])
        for item in items:
            title = item.get('title', '').strip()
            if not title or len(title) < 5:
                continue
            title_key = re.sub(r'[^\w가-힣]', '', title.lower())[:20]
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            result.append({
                'icon':    icon_map.get(section_key, '📌'),
                'title':   title,
                'body':    item.get('body', '') or item.get('summary_ko', '') or '',
                'url':     item.get('url', ''),
                'source':  item.get('source', ''),
                'section': section_key,
            })
            if len(result) >= max_items:
                return result
    return result[:max_items]


# ── LLM 압축 ─────────────────────────────────────────────────

def compress_news_for_card(
    client: OpenAI, category: str, news_list: list, summary: list
) -> dict:
    if not news_list:
        return {'news': [], 'insight': ''}

    news_text    = ""
    summary_text = '\n'.join(f"- {s}" for s in summary) if summary else "없음"

    for i, n in enumerate(news_list, 1):
        body = (n.get('body', '') or '')[:300]
        news_text += f"\n{i}. 제목: {n['title']}\n   내용: {body}\n"

    prompt = f"""당신은 [{category}] 분야 IT 트렌드를 실무자에게 전달하는 에디터입니다.

핵심 요약:
{summary_text}

뉴스 목록:
{news_text}

아래 JSON 형식으로 작성하세요.

headline 규칙:
- 30자 이내, 숫자/비교/결과로 임팩트 있게
- 예: "딥시크 API 가격 75% 영구 인하"
- 예: "MS 파라1.5, 브라우저 직접 조작 에이전트"

summary 규칙:
- 50자 이내, 왜 중요한지 자연스러운 한국어 1문장
- 예: "비용 절감으로 중소 개발팀도 대형 모델 도입 현실화"

insight 규칙:
- 60자 이내, 이 분야 실무자가 이번 주 바로 적용할 수 있는 액션 1문장
- 자연스러운 구어체 한국어로 작성
- 예: "에이전트 파이프라인에 검증 게이트를 먼저 설계하고 배포하세요"
- 금지: why:, apply:, (1), (2) 같은 구분자 사용
- 금지: 대괄호 [] 사용

JSON:
{{
  "news": [
    {{"idx": 1, "headline": "...", "summary": "..."}},
    {{"idx": 2, "headline": "...", "summary": "..."}},
    {{"idx": 3, "headline": "...", "summary": "..."}},
    {{"idx": 4, "headline": "...", "summary": "..."}},
    {{"idx": 5, "headline": "...", "summary": "..."}}
  ],
  "insight": "자연스러운 1문장 액션"
}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw      = json.loads(resp.choices[0].message.content)
        news_map = {item['idx']: item for item in raw.get('news', [])}

        for i, n in enumerate(news_list, 1):
            mapped        = news_map.get(i, {})
            n['headline'] = truncate(mapped.get('headline', n['title'][:30]), 35)
            n['summary']  = truncate(mapped.get('summary', ''), 60)

        # insight 정제 - 구분자/대괄호 제거
        insight = raw.get('insight', '')
        if isinstance(insight, list):
            insight = ' '.join(str(x) for x in insight if x)
        insight = str(insight).strip()
        insight = re.sub(r'\b(why|apply|why:|apply:)\b\s*', '', insight, flags=re.IGNORECASE)
        insight = re.sub(r'^\[.*?\]\s*', '', insight)
        insight = re.sub(r'\|.*', '', insight)        # | 이후 제거
        insight = re.sub(r';\s*apply.*', '', insight, flags=re.IGNORECASE)
        insight = re.sub(r'\s+', ' ', insight).strip()
        insight = truncate(insight, 70)

        return {'news': news_list, 'insight': insight}

    except Exception as e:
        print(f"  뉴스 LLM 압축 실패 ({category}): {e}")
        for n in news_list:
            n['headline'] = n['title'][:30]
            n['summary']  = ''
        return {'news': news_list, 'insight': ''}


def compress_papers_for_card(
    client: OpenAI, papers_by_cat: dict
) -> dict:
    if not any(papers_by_cat.values()):
        return {}

    papers_text = ""
    for cat, papers in papers_by_cat.items():
        if not papers:
            continue
        papers_text += f"\n[{cat}]\n"
        for i, p in enumerate(papers[:5], 1):
            abstract = (p.get('abstract', '') or '')[:300]
            url      = p.get('url', '')
            papers_text += f"  {i}. {p['title']}\n     URL: {url}\n     {abstract}\n"

    prompt = f"""다음 arXiv 논문에서 실무자에게 유용한 것을 선별해 한국어로 요약하세요.

{papers_text}

각 카테고리에서 2편씩 선별하고 JSON으로 작성하세요.

규칙:
- title_ko: 논문 제목 한국어 번역 (25자 이내, 간결하게)
- why: 왜 지금 주목해야 하는지 (40자 이내, 핵심만)
- apply: 실무에서 어떻게 써먹을 수 있는지 (40자 이내, 구체적으로)
- url: 논문 URL 그대로 복사

JSON:
{{
  "AI": [
    {{"title_ko": "...", "why": "...", "apply": "...", "url": "..."}}
  ],
  "데이터엔지니어링": [
    {{"title_ko": "...", "why": "...", "apply": "...", "url": "..."}}
  ],
  "RPA": [
    {{"title_ko": "...", "why": "...", "apply": "...", "url": "..."}}
  ]
}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_QUALITY,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  논문 LLM 압축 실패: {e}")
        return {}


# ── 색상/레이블 상수 ──────────────────────────────────────────

COLOR_MAP = {
    'ai':    {'main': 'var(--ai)',    'bg': 'rgba(123,110,246,0.12)',
              'grad': 'linear-gradient(90deg,var(--ai),#a78bfa)'},
    'de':    {'main': 'var(--de)',    'bg': 'rgba(0,201,167,0.12)',
              'grad': 'linear-gradient(90deg,var(--de),#34d8b5)'},
    'rpa':   {'main': 'var(--rpa)',   'bg': 'rgba(255,107,107,0.12)',
              'grad': 'linear-gradient(90deg,var(--rpa),#ff9a9a)'},
    'paper': {'main': 'var(--paper)', 'bg': 'rgba(167,139,250,0.12)',
              'grad': 'linear-gradient(90deg,var(--paper),#c4b5fd)'},
}
CAT_LABEL = {
    'AI': 'Artificial Intelligence',
    '데이터엔지니어링': 'Data Engineering',
    'RPA': 'RPA & Automation',
}
CAT_COLOR    = {'AI': 'ai', '데이터엔지니어링': 'de', 'RPA': 'rpa'}
CAT_HEADLINE = {'AI': 'AI', '데이터엔지니어링': 'DATA', 'RPA': 'RPA'}
CAT_PAPER_COLOR = {
    'AI':            ('var(--ai)',    'rgba(123,110,246,0.10)'),
    '데이터엔지니어링': ('var(--de)',    'rgba(0,201,167,0.10)'),
    'RPA':           ('var(--rpa)',   'rgba(255,107,107,0.10)'),
}


def _footer(slide_num: int, total: int) -> str:
    now         = datetime.now(KST)
    week_num    = (now.day - 1) // 7 + 1
    footer_week = f"{now.year}.{now.month:02d} W{week_num}"
    return f'''
    <div class="slide-footer">
      <span class="footer-week">IT WEEKLY · {footer_week}</span>
      <span class="footer-num">0{slide_num} / 0{total}</span>
    </div>'''


# ── 슬라이드 빌더 ────────────────────────────────────────────

def build_cover_slide(week_label: str, total: int) -> str:
    now      = datetime.now(KST)
    week_num = (now.day - 1) // 7 + 1
    date_str = f"{now.year}.{now.month:02d}.{now.day:02d}"
    week_str = f"{now.year} MAY WEEK {week_num}"

    return f'''
<div class="slide-label">SLIDE 01 / 0{total} — 표지</div>
<div class="slide cover-slide">
  <div class="grid-bg"></div>
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="cover-inner">
    <div class="cover-badge">WEEKLY TREND REPORT</div>
    <div>
      <div class="cover-title">
        IT<br>
        <span class="cover-title-grad">TREND</span><br>
        DIGEST
      </div>
      <div class="cover-desc">
        매주 놓치면 안 될
        <strong>AI · 데이터엔지니어링 · RPA</strong><br>
        핵심 동향을 한 장으로 정리합니다.
      </div>
    </div>
    <div class="cover-bottom">
      <div class="cover-tags">
        <span class="tag tag-ai">AI</span>
        <span class="tag tag-de">DATA ENG</span>
        <span class="tag tag-rpa">RPA</span>
        <span class="tag tag-paper">RESEARCH</span>
      </div>
      <div class="cover-date">{date_str}<br>{week_str}</div>
    </div>
  </div>
</div>'''


def build_category_slide(
    cat: str, slide_num: int, total: int,
    summary: list, compressed: dict,
) -> str:
    color     = CAT_COLOR.get(cat, 'ai')
    c         = COLOR_MAP[color]
    headline  = CAT_HEADLINE[cat]
    cat_label = CAT_LABEL.get(cat, cat)

    news_list    = compressed.get('news', [])
    insight      = compressed.get('insight', '')
    summary_text = truncate(summary[0], 80) if summary else ''

    # 뉴스 아이템 HTML (제목 링크 + 요약)
    news_html = ''
    for news in news_list[:5]:
        hl   = news.get('headline', news.get('title', ''))
        sm   = news.get('summary', '')
        icon = news.get('icon', '📌')
        url  = news.get('url', '')

        # 제목에 URL 연결 (있을 경우)
        if url:
            title_html = f'<a class="news-link" href="{url}" title="{url}">{hl}</a>'
        else:
            title_html = f'<span>{hl}</span>'

        news_html += f'''
      <div class="news-item">
        <span class="news-icon">{icon}</span>
        <div class="news-body">
          <div class="news-hl">{title_html}</div>
          {f'<div class="news-sm">{sm}</div>' if sm else ''}
        </div>
      </div>'''

    if not news_html:
        news_html = '<div class="empty-msg">수집된 뉴스가 없습니다.</div>'

    insight_html = f'''
    <div class="insight-box">
      <span class="insight-icon">💡</span>
      <span class="insight-text">{insight}</span>
    </div>''' if insight else ''

    return f'''
<div class="slide-label">SLIDE 0{slide_num} / 0{total} — {cat}</div>
<div class="slide cat-slide" style="--c-main:{c["main"]};--c-bg:{c["bg"]};--c-grad:{c["grad"]};">
  <div class="cat-topbar"></div>
  <div class="cat-blob"></div>
  <div class="cat-inner">
    <div class="cat-header">
      <div class="cat-dot"></div>
      <span class="cat-label-text">{cat_label}</span>
      <span class="cat-num">0{slide_num} / 0{total}</span>
    </div>
    <div class="cat-title">{headline}<br>WEEKLY</div>
    <div class="cat-summary">
      <div class="cat-summary-text">{summary_text}</div>
    </div>
    <div class="news-list">{news_html}</div>
    {insight_html}
    {_footer(slide_num, total)}
  </div>
</div>'''


def build_paper_slide(
    slide_num: int, total: int, papers_compressed: dict
) -> str:
    c = COLOR_MAP['paper']

    cat_colors = {
        'AI':            ('var(--ai)',    'rgba(123,110,246,0.10)'),
        '데이터엔지니어링': ('var(--de)',    'rgba(0,201,167,0.10)'),
        'RPA':           ('var(--rpa)',   'rgba(255,107,107,0.10)'),
    }

    # 내용 있는 카테고리만 행으로 표시
    rows_html = ''
    for cat in ['AI', '데이터엔지니어링', 'RPA']:
        papers       = papers_compressed.get(cat, [])
        if not papers:
            continue
        c_main, c_bg = cat_colors.get(cat, ('var(--paper)', 'rgba(167,139,250,0.10)'))

        items_html = ''
        for p in papers[:2]:
            title_ko = truncate(p.get('title_ko', ''), 30)
            why      = truncate(p.get('why', ''), 50)
            apply    = truncate(p.get('apply', ''), 50)
            url      = p.get('url', '')

            title_html = (
                f'<a class="paper-link" href="{url}" title="{url}">{title_ko}</a>'
                if url else f'<span>{title_ko}</span>'
            )

            items_html += f'''
            <div class="paper-item" style="border-left-color:{c_main};background:{c_bg};">
              <div class="paper-title">{title_html}</div>
              {f'<div class="paper-why">📌 {why}</div>' if why else ''}
              {f'<div class="paper-apply">💼 {apply}</div>' if apply else ''}
            </div>'''

        rows_html += f'''
      <div class="paper-row">
        <div class="paper-row-label" style="color:{c_main};">{cat}</div>
        <div class="paper-row-items">{items_html}</div>
      </div>'''

    if not rows_html:
        rows_html = '<div class="empty-msg">수집된 논문이 없습니다.</div>'

    return f'''
<div class="slide-label">SLIDE 0{slide_num} / 0{total} — 연구 동향</div>
<div class="slide paper-slide" style="--c-main:{c["main"]};--c-grad:{c["grad"]};">
  <div class="cat-topbar"></div>
  <div class="cat-inner">
    <div class="cat-header">
      <div class="cat-dot"></div>
      <span class="cat-label-text">arXiv Research</span>
      <span class="cat-num">0{slide_num} / 0{total}</span>
    </div>
    <div class="cat-title">RESEARCH<br>DIGEST</div>
    <div class="paper-grid">
      {rows_html}
    </div>
    {_footer(slide_num, total)}
  </div>
</div>'''


def build_closing_slide(
    slide_num: int, total: int, all_insights: list
) -> str:
    actions_html = ''
    for i, action in enumerate(all_insights[:3], 1):
        text = action['text']
        text = re.sub(r'\b(why:|apply:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\|.*', '', text)
        text = truncate(text.strip(), 60)

        actions_html += f'''
      <li class="action-item">
        <span class="action-num">0{i}</span>
        <div class="action-body">
          <div class="action-cat">[{action["cat"]}]</div>
          <div class="action-text">{text}</div>
        </div>
      </li>'''

    return f'''
<div class="slide-label">SLIDE 0{slide_num} / 0{total} — 마무리</div>
<div class="slide closing-slide">
  <div class="grid-bg grid-bg-dim"></div>
  <div class="closing-bar"></div>
  <div class="closing-inner">
    <div class="closing-top">
      <div class="closing-label">THIS WEEK\'S ACTION ITEMS</div>
      <div class="closing-title">지금 바로<br>챙겨야 할<br>3가지</div>
      <ul class="action-list">{actions_html}</ul>
    </div>
    <div class="closing-footer">
      <div class="closing-brand">TREND DIGEST</div>
      <div class="closing-cta">
        매주 월요일 업데이트<br>
        <strong>팔로우하고 놓치지 마세요 →</strong>
      </div>
    </div>
  </div>
</div>'''


# ── CSS ──────────────────────────────────────────────────────

def get_css() -> str:
    return '''
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Bebas+Neue&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --ai:#7B6EF6; --de:#00C9A7; --rpa:#FF6B6B; --paper:#a78bfa;
    --dark:#0D0D14; --card-bg:#111118;
    --text:#F0F0F8; --muted:#8888AA; --accent:#F5C842;
    --border:rgba(255,255,255,0.07);
  }
  * { margin:0; padding:0; box-sizing:border-box; }

  @page { size: A4; margin: 0; }

  body {
    font-family:'Noto Sans KR',sans-serif;
    background:#1a1a2e;
    display:flex; flex-direction:column; align-items:center;
    padding:40px 20px; gap:28px;
  }

  .slide {
    width:794px; height:1123px;
    position:relative; overflow:hidden; flex-shrink:0;
  }
  .slide-label {
    color:#555577; font-family:'JetBrains Mono',monospace;
    font-size:12px; text-align:center; margin-bottom:6px; letter-spacing:2px;
  }

  @media print {
    body { background:white; padding:0; gap:0; }
    .slide-label { display:none; }
    .slide { page-break-after:always; }
  }

  /* ── 공통 ── */
  .grid-bg {
    position:absolute; inset:0;
    background-image:
      linear-gradient(rgba(123,110,246,0.06) 1px,transparent 1px),
      linear-gradient(90deg,rgba(123,110,246,0.06) 1px,transparent 1px);
    background-size:52px 52px;
  }
  .grid-bg-dim {
    background-image:
      linear-gradient(rgba(255,255,255,0.02) 1px,transparent 1px),
      linear-gradient(90deg,rgba(255,255,255,0.02) 1px,transparent 1px);
  }
  .cat-topbar {
    position:absolute; top:0; left:0; right:0; height:5px;
    background:var(--c-grad);
  }
  .cat-blob {
    position:absolute; width:400px; height:400px; border-radius:50%;
    background:radial-gradient(circle,var(--c-bg) 0%,transparent 70%);
    top:-80px; right:-80px;
  }
  .cat-inner {
    background:#1a1a2e;
    position:relative; z-index:2; height:100%;
    padding:40px 60px 32px;
    display:flex; flex-direction:column;
  }
  .cat-header {
    display:flex; align-items:center; gap:12px;
    margin-bottom:14px; flex-shrink:0;
  }
  .cat-dot {
    width:12px; height:12px; border-radius:50%;
    background:var(--c-main); flex-shrink:0;
  }
  .cat-label-text {
    font-family:'JetBrains Mono',monospace; font-size:14px;
    letter-spacing:3px; text-transform:uppercase; color:var(--c-main);
  }
  .cat-num {
    margin-left:auto; font-family:'Bebas Neue',sans-serif;
    font-size:20px; letter-spacing:2px; opacity:0.18; color:#fff;
  }
  .cat-title {
    font-family:'Bebas Neue',sans-serif; font-size:72px;
    line-height:1; letter-spacing:3px; color:#fff;
    margin-bottom:14px; flex-shrink:0;
  }
  .cat-summary {
    background:rgba(255,255,255,0.05);
    border-left:4px solid var(--c-main);
    padding:14px 18px; margin-bottom:16px;
    border-radius:0 3px 3px 0; flex-shrink:0;
  }
  .cat-summary-text {
    font-size:17px; line-height:1.6; color:var(--text); word-break:keep-all;
  }

  /* ── 뉴스: 남은 공간 균등 분배 ── */
  .news-list {
    flex:1; display:flex; flex-direction:column;
    justify-content:space-between;
    min-height:0;
  }
  .news-item {
    display:flex; gap:14px;
    padding:0;
    border-bottom:1px solid var(--border);
    align-items:flex-start;
    flex:1;
    align-content:center;
    padding:10px 0;
  }
  .news-item:last-child { border-bottom:none; }
  .news-icon { font-size:20px; flex-shrink:0; margin-top:2px; }
  .news-body { flex:1; }
  .news-hl {
    font-size:18px; font-weight:700; line-height:1.4;
    word-break:keep-all; color:#fff;
  }
  .news-link {
    color:#fff; text-decoration:none;
    border-bottom:1px solid rgba(255,255,255,0.15);
    transition:border-color 0.15s;
  }
  .news-link:hover { border-bottom-color:var(--c-main); color:var(--c-main); }
  .news-sm {
    font-size:14px; color:var(--muted); line-height:1.5;
    margin-top:4px; word-break:keep-all;
  }
  .empty-msg { font-size:15px; color:var(--muted); padding:12px 0; }

  /* ── 인사이트 ── */
  .insight-box {
    margin-top:14px; padding:16px 20px; flex-shrink:0;
    background:rgba(245,200,66,0.07);
    border:1px solid rgba(245,200,66,0.22);
    border-radius:3px;
    display:flex; gap:10px; align-items:flex-start;
  }
  .insight-icon { font-size:16px; flex-shrink:0; padding-top:2px; }
  .insight-text {
    font-size:16px; color:var(--text); line-height:1.6; word-break:keep-all;
  }

  /* ── 푸터 ── */
  .slide-footer {
    display:flex; justify-content:space-between; align-items:center;
    margin-top:14px; flex-shrink:0;
  }
  .footer-week {
    font-family:'JetBrains Mono',monospace; font-size:12px;
    letter-spacing:2px; color:var(--muted);
  }
  .footer-num {
    font-family:'Bebas Neue',sans-serif; font-size:18px;
    letter-spacing:2px; color:rgba(255,255,255,0.1);
  }

  /* ── 표지 ── */
  .cover-slide { background:var(--dark); }
  .blob { position:absolute; border-radius:50%; }
  .blob-1 {
    width:560px; height:560px; top:-120px; left:-120px;
    background:radial-gradient(circle,rgba(123,110,246,0.15) 0%,transparent 65%);
  }
  .blob-2 {
    width:420px; height:420px; bottom:-80px; right:-80px;
    background:radial-gradient(circle,rgba(0,201,167,0.10) 0%,transparent 65%);
  }
  .cover-inner {
    position:relative; z-index:2; height:100%;
    display:flex; flex-direction:column;
    justify-content:space-between; padding:70px 72px;
  }
  .cover-badge {
    display:inline-flex; align-items:center;
    background:rgba(245,200,66,0.10); border:1px solid rgba(245,200,66,0.25);
    color:var(--accent); font-family:'JetBrains Mono',monospace;
    font-size:15px; letter-spacing:3px; padding:10px 20px; width:fit-content;
  }
  .cover-title {
    font-family:'Bebas Neue',sans-serif; font-size:130px;
    line-height:0.90; letter-spacing:6px; color:#fff; margin-top:24px;
  }
  .cover-title-grad {
    background:linear-gradient(135deg,var(--ai),var(--de));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
  }
  .cover-desc {
    font-size:22px; color:var(--muted); font-weight:300;
    margin-top:20px; line-height:1.7;
  }
  .cover-desc strong { color:var(--text); font-weight:500; }
  .cover-bottom { display:flex; justify-content:space-between; align-items:flex-end; }
  .cover-tags { display:flex; gap:10px; }
  .tag {
    padding:8px 18px; border-radius:2px;
    font-size:14px; font-weight:700; letter-spacing:1px;
  }
  .tag-ai    { background:rgba(123,110,246,0.18); color:var(--ai);    border:1px solid rgba(123,110,246,0.35); }
  .tag-de    { background:rgba(0,201,167,0.18);   color:var(--de);    border:1px solid rgba(0,201,167,0.35); }
  .tag-rpa   { background:rgba(255,107,107,0.18); color:var(--rpa);   border:1px solid rgba(255,107,107,0.35); }
  .tag-paper { background:rgba(167,139,250,0.18); color:var(--paper); border:1px solid rgba(167,139,250,0.35); }
  .cover-date {
    text-align:right; font-family:'JetBrains Mono',monospace;
    font-size:14px; color:var(--muted); letter-spacing:1px; line-height:1.9;
  }

  /* ── 논문 슬라이드: 내용 있는 열만 표시, 균등 분배 ── */
  .paper-slide { background:var(--card-bg); }
  .paper-grid {
    flex:1; display:flex; flex-direction:column;
    gap:0; min-height:0; justify-content:space-between;
  }
  /* 카테고리 행: 가로로 라벨+카드 배치 */
  .paper-row {
    display:flex; gap:16px; align-items:flex-start;
    padding:16px 0; border-bottom:1px solid var(--border);
    flex:1;
  }
  .paper-row:last-child { border-bottom:none; }
  .paper-row-label {
    font-family:'JetBrains Mono',monospace; font-size:12px;
    letter-spacing:2px; text-transform:uppercase;
    flex-shrink:0; width:80px; padding-top:4px;
  }
  .paper-row-items {
    flex:1; display:flex; gap:12px;
  }
  .paper-item {
    flex:1; padding:14px 16px; border-left:3px solid;
    border-radius:0 4px 4px 0;
  }
  .paper-title {
    font-size:15px; font-weight:700; color:#fff;
    line-height:1.4; margin-bottom:8px; word-break:keep-all;
  }
  .paper-link {
    color:#fff; text-decoration:none;
    border-bottom:1px solid rgba(255,255,255,0.15);
  }
  .paper-link:hover { color:var(--paper); border-bottom-color:var(--paper); }
  .paper-why {
    font-size:13px; color:var(--muted); line-height:1.5;
    margin-bottom:6px; word-break:keep-all;
  }
  .paper-apply {
    font-size:13px; color:var(--accent); line-height:1.5; word-break:keep-all;
  }

  /* ── 마무리: 액션 아이템이 화면 채우도록 ── */
  .closing-slide { background:var(--dark); }
  .closing-bar {
    position:absolute; bottom:0; left:0; right:0; height:6px;
    background:linear-gradient(90deg,var(--ai) 0%,var(--de) 50%,var(--rpa) 100%);
  }
  .closing-inner {
    position:relative; z-index:2; height:100%;
    padding:60px 68px 70px;
    display:flex; flex-direction:column; justify-content:space-between;
  }
  .closing-top { display:flex; flex-direction:column; flex:1; }
  .closing-label {
    font-family:'JetBrains Mono',monospace; font-size:14px;
    letter-spacing:4px; color:var(--accent); margin-bottom:16px;
    flex-shrink:0;
  }
  .closing-title {
    font-family:'Bebas Neue',sans-serif; font-size:100px;
    line-height:0.88; letter-spacing:4px; color:#fff; margin-bottom:36px;
    flex-shrink:0;
  }
  .action-list {
    list-style:none; display:flex; flex-direction:column;
    flex:1; justify-content:space-between;
  }
  .action-item {
    display:flex; gap:20px; padding:20px 24px;
    background:rgba(255,255,255,0.04); border-radius:3px;
    align-items:center;
  }
  .action-num {
    font-family:'Bebas Neue',sans-serif; font-size:48px;
    line-height:1; color:var(--accent); flex-shrink:0; min-width:44px;
  }
  .action-body { padding-top:0; }
  .action-cat {
    font-family:'JetBrains Mono',monospace; font-size:12px;
    color:var(--accent); letter-spacing:2px; margin-bottom:6px;
  }
  .action-text { font-size:20px; color:var(--text); line-height:1.5; word-break:keep-all; }
  .closing-footer { display:flex; justify-content:space-between; align-items:flex-end; flex-shrink:0; }
  .closing-brand {
    font-family:'Bebas Neue',sans-serif; font-size:28px;
    letter-spacing:4px; color:rgba(255,255,255,0.10);
  }
  .closing-cta { text-align:right; font-size:16px; color:var(--muted); line-height:1.7; }
  .closing-cta strong { color:var(--accent); font-size:19px; }
'''


# ── 전체 HTML ────────────────────────────────────────────────

def generate_card_html(
    reports: dict, week_label: str,
    client: OpenAI, supabase: Client
) -> str:
    TOTAL = 6

    print("  논문 수집 중...")
    papers_by_cat     = fetch_papers(supabase, days=7)
    has_papers        = any(papers_by_cat.values())
    print("  논문 LLM 압축 중...")
    papers_compressed = compress_papers_for_card(client, papers_by_cat) if has_papers else {}

    categories = [
        ('AI',            2),
        ('데이터엔지니어링', 3),
        ('RPA',           4),
    ]

    category_slides = ''
    all_insights    = []

    for cat, slide_num in categories:
        data          = reports.get(cat, {'report_text': '', 'sections_json': {}})
        report_text   = data.get('report_text', '')
        sections_json = data.get('sections_json', {})

        summary   = parse_summary(report_text)
        news_list = extract_news_from_sections(sections_json, max_items=5)

        print(f"  [{cat}] 뉴스 {len(news_list)}개, LLM 압축 중...")
        compressed = compress_news_for_card(client, cat, news_list, summary)

        category_slides += build_category_slide(
            cat, slide_num, TOTAL, summary, compressed
        )
        if compressed.get('insight'):
            all_insights.append({'cat': cat, 'text': compressed['insight']})

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>IT 주간 동향 카드뉴스 | {week_label}</title>
<style>
{get_css()}
</style>
</head>
<body>
{build_cover_slide(week_label, TOTAL)}
{category_slides}
{build_paper_slide(5, TOTAL, papers_compressed)}
{build_closing_slide(6, TOTAL, all_insights)}
</body>
</html>'''


# ── 메인 ────────────────────────────────────────────────────

def main():
    supabase_url   = os.getenv('SUPABASE_URL')
    supabase_key   = os.getenv('SUPABASE_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not all([supabase_url, supabase_key, openai_api_key]):
        raise ValueError("환경변수 누락: SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY 확인")

    supabase   = create_client(supabase_url, supabase_key)
    client     = OpenAI(api_key=openai_api_key)
    week_label = get_week_label()
    print(f"카드뉴스 생성: {week_label}")

    reports = fetch_reports(supabase, week_label)
    if not reports:
        print(f"❌ {week_label} 리포트가 DB에 없습니다.")
        return
    print(f"✅ {list(reports.keys())} 카테고리 로드 완료")

    html = generate_card_html(reports, week_label, client, supabase)

    filename = (
        f"linkedin_card_{week_label}"
        .replace(' ', '_')
        .replace('년', '').replace('월', '').replace('주차', 'w')
        + '.html'
    )
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ 저장 완료: {output}")
    print(f"\n📌 PDF 변환:")
    print(f"   1. Chrome에서 {filename} 열기")
    print(f"   2. Cmd+P → 용지: A4 / 여백: 없음 / 배경 그래픽: 체크")
    print(f"   3. PDF로 저장")
    print(f"   4. 링크드인 → 문서 업로드 → PDF 선택")


if __name__ == "__main__":
    main()