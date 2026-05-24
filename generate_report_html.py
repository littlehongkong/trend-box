# generate_report_html.py
# 변경사항:
#   1. Google News 섹션 (📡 추가 뉴스) 화면에 표시
#   2. 뉴스 제목 하단 핵심 요약 표시 + 로컬 본문 파일 연결
#   3. 논문 초록 핵심 요약 + 로컬 파일 연결
#   4. articles/, papers/ 폴더에 로컬 파일 저장
#   5. ZIP 패키징 (폐쇄망 배포용)

import os
import re
import json
import zipfile
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

KST = timezone(timedelta(hours=9))

SECTION_LABELS = {
    "new_services":   "🚀 New Services & Launches",
    "updates":        "🔄 Updates & Policy Changes",
    "investment":     "💰 Investment & Business",
    "infrastructure": "🛠 Infrastructure & Dev Tools",
    "trends":         "📊 Technology Trends & Research",
}


# ── 유틸 ────────────────────────────────────────────────────────

def get_week_label() -> str:
    now      = datetime.now(KST)
    week_num = (now.day - 1) // 7 + 1
    return f"{now.year}년 {now.month}월 {week_num}주차"


def safe_filename(text: str, max_len: int = 40) -> str:
    """파일명에 사용 가능한 문자열로 변환"""
    text = re.sub(r'[\\/:*?"<>|]', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text[:max_len]


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


def parse_summary_only(report_text: str) -> list:
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


def parse_actions_only(report_text: str) -> list:
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
            text = re.sub(r'\s*https?://\S+', '', text)
            text = text.strip().lstrip('.').strip()
            text = re.sub(r'\([^)]*$', '', text).strip()
            if text and len(text) > 10:
                actions.append(text)
    return actions[:3]


def parse_research_from_report(report_text: str) -> list:
    """리포트 텍스트에서 논문 섹션 파싱"""
    items       = []
    in_research = False
    current     = None

    for line in report_text.split('\n'):
        line = line.strip()

        if '연구 동향' in line or 'Research Trend' in line or '📚' in line:
            in_research = True; continue

        if in_research and line.startswith('##') and '연구' not in line:
            break

        if in_research:
            # 논문 제목 패턴: - **[제목]** 또는 - **제목**
            m = re.search(r'\*\*(.*?)\*\*', line)
            if m and (line.startswith('-') or line.startswith('*')):
                title = m.group(1).strip('[]')
                rest  = line[m.end():].lstrip(' —-').strip()

                url_m = re.search(r'https?://\S+', rest)
                url   = url_m.group(0).rstrip(')') if url_m else ''
                body  = re.sub(r'https?://\S+', '', rest).strip().rstrip('()')

                current = {'title': title, 'body': body[:300], 'url': url, 'detail': ''}
                items.append(current)

            elif current and line.startswith('-') and current:
                # 핵심기여, 실무관련성 등 세부 항목
                detail = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                detail = re.sub(r'^[-•]\s*', '', detail).strip()
                url_m  = re.search(r'https?://\S+', detail)
                if url_m and not current['url']:
                    current['url'] = url_m.group(0).rstrip(')')
                elif detail and not detail.startswith('http'):
                    current['detail'] += detail + ' '

    return items[:5]


# ── 로컬 파일 저장 ───────────────────────────────────────────────

def save_article_file(
    item: dict, idx: int, category: str, output_dir: str
) -> str:
    """뉴스 기사 본문을 로컬 HTML 파일로 저장, 상대경로 반환"""
    articles_dir = os.path.join(output_dir, 'articles')
    os.makedirs(articles_dir, exist_ok=True)

    title    = item.get('title', f'article_{idx}')
    body     = item.get('body', '') or item.get('summary_ko', '') or ''
    url      = item.get('url', '')
    source   = item.get('source', '')
    filename = f"{category}_{idx:02d}_{safe_filename(title)}.html"
    filepath = os.path.join(articles_dir, filename)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: 'Noto Sans KR', sans-serif; max-width: 800px;
    margin: 40px auto; padding: 0 20px; line-height: 1.8;
    background: #fafafa; color: #333; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #6c63ff;
    padding-bottom: 12px; margin-bottom: 24px; }}
  .meta {{ font-size: 13px; color: #888; margin-bottom: 24px; }}
  .body {{ font-size: 15px; background: #fff; padding: 24px;
    border-radius: 4px; border: 1px solid #eee; }}
  .back {{ margin-top: 32px; }}
  .back a {{ color: #6c63ff; text-decoration: none; font-size: 14px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">
    출처: {source}
    {f' | <a href="{url}" target="_blank">원문 링크</a>' if url else ''}
  </div>
  <div class="body">{body if body else '본문 내용이 없습니다.'}</div>
  <div class="back"><a href="../weekly_report.html">← 리포트로 돌아가기</a></div>
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    return f"articles/{filename}"


def save_paper_file(
    item: dict, idx: int, category: str, output_dir: str
) -> str:
    """논문 초록을 로컬 HTML 파일로 저장, 상대경로 반환"""
    papers_dir = os.path.join(output_dir, 'papers')
    os.makedirs(papers_dir, exist_ok=True)

    title    = item.get('title', f'paper_{idx}')
    body     = item.get('body', '') or ''
    detail   = item.get('detail', '') or ''
    url      = item.get('url', '')
    filename = f"{category}_{idx:02d}_{safe_filename(title)}.html"
    filepath = os.path.join(papers_dir, filename)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: 'Noto Sans KR', sans-serif; max-width: 800px;
    margin: 40px auto; padding: 0 20px; line-height: 1.8;
    background: #fafafa; color: #333; }}
  h1 {{ font-size: 20px; border-bottom: 2px solid #a78bfa;
    padding-bottom: 12px; margin-bottom: 24px; }}
  .badge {{ display: inline-block; background: rgba(167,139,250,0.15);
    color: #a78bfa; font-size: 12px; padding: 2px 10px;
    border-radius: 2px; margin-bottom: 16px; }}
  .section {{ margin-bottom: 20px; }}
  .section h3 {{ font-size: 14px; color: #666; margin-bottom: 8px; }}
  .section p {{ font-size: 15px; background: #fff; padding: 16px;
    border-radius: 4px; border: 1px solid #eee; }}
  .back {{ margin-top: 32px; }}
  .back a {{ color: #a78bfa; text-decoration: none; font-size: 14px; }}
</style>
</head>
<body>
  <div class="badge">arXiv · {category}</div>
  <h1>{title}</h1>
  {f'<div class="section"><h3>📝 핵심 내용</h3><p>{body}</p></div>' if body else ''}
  {f'<div class="section"><h3>💡 실무 관련성</h3><p>{detail}</p></div>' if detail else ''}
  {f'<div class="section"><h3>🔗 arXiv 링크</h3><p><a href="{url}">{url}</a></p></div>' if url else ''}
  <div class="back"><a href="../weekly_report.html">← 리포트로 돌아가기</a></div>
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    return f"papers/{filename}"


# ── HTML 섹션 빌더 ──────────────────────────────────────────────

def build_news_item_html(
    item: dict, local_path: str = '', source_type: str = 'domestic'
) -> str:
    """뉴스 아이템 HTML (제목 + 핵심요약 + 로컬링크)"""
    title      = item.get('title', '')
    body       = item.get('body', '') or item.get('summary_ko', '') or ''
    source     = item.get('source', '')
    url        = item.get('url', '')

    # 요약 (100자)
    summary = body[:120].rstrip() + '...' if len(body) > 120 else body

    badge_color = {
        'domestic': 'domestic',
        'global':   'global',
        'google':   'google',
    }.get(source_type, 'domestic')

    detail_btn = (
        f'<a class="detail-link" href="{local_path}">자세히 보기 →</a>'
        if local_path else ''
    )
    url_text = (
        f'<span class="news-url">{url}</span>'
        if url else ''
    )

    return f'''
    <div class="news-item">
      <div class="news-item-header">
        <div class="news-title">{title}</div>
        <span class="source-badge {badge_color}">{source}</span>
      </div>
      {f'<div class="news-summary">{summary}</div>' if summary else ''}
      <div class="news-footer">
        {url_text}
        {detail_btn}
      </div>
    </div>'''


def build_paper_item_html(item: dict, local_path: str = '') -> str:
    """논문 아이템 HTML (제목 + 초록요약 + 실무관련성 + 로컬링크)"""
    title  = item.get('title', '')
    body   = item.get('body', '')
    detail = item.get('detail', '').strip()
    url    = item.get('url', '')

    summary = body[:150].rstrip() + '...' if len(body) > 150 else body

    detail_btn = (
        f'<a class="detail-link arxiv-link" href="{local_path}">초록 전문 →</a>'
        if local_path else ''
    )

    return f'''
    <div class="news-item paper-item">
      <div class="news-item-header">
        <div class="news-title">{title}</div>
        <span class="source-badge arxiv">arXiv</span>
      </div>
      {f'<div class="news-summary">{summary}</div>' if summary else ''}
      {f'<div class="paper-insight">💡 {detail}</div>' if detail else ''}
      <div class="news-footer">
        {f'<span class="news-url">{url}</span>' if url else ''}
        {detail_btn}
      </div>
    </div>'''


def build_tab_html(
    category: str, color_var: str, data: dict,
    tab_num: int, output_dir: str
) -> str:
    """탭 콘텐츠 HTML 생성"""

    report_text   = data.get('report_text', '')
    sections_json = data.get('sections_json', {})

    # ── 핵심 요약 ────────────────────────────────────────────
    summary      = parse_summary_only(report_text)
    summary_html = '\n'.join([f'<li>{s}</li>' for s in summary]) \
                   if summary else '<li>요약 정보가 없습니다.</li>'

    # ── 국내 주요 뉴스 ────────────────────────────────────────
    domestic_items = sections_json.get('domestic', [])
    domestic_html  = ''
    for i, item in enumerate(domestic_items):
        local_path = save_article_file(item, i+1, f'{category}_domestic', output_dir)
        domestic_html += build_news_item_html(item, local_path, 'domestic')
    if not domestic_html:
        domestic_html = '<div class="empty-msg">수집된 국내 뉴스가 없습니다.</div>'

    # ── 글로벌 동향 ───────────────────────────────────────────
    global_items = sections_json.get('global', [])
    global_html  = ''
    for i, item in enumerate(global_items):
        local_path = save_article_file(item, i+1, f'{category}_global', output_dir)
        global_html += build_news_item_html(item, local_path, 'global')
    if not global_html:
        global_html = '<div class="empty-msg">수집된 글로벌 뉴스가 없습니다.</div>'

    # ── 추가 뉴스 (Google News 섹션) ─────────────────────────
    google_html  = ''
    google_count = 0
    for section_key in ['new_services', 'updates', 'investment', 'infrastructure', 'trends']:
        items = sections_json.get(section_key, [])
        if not items: continue
        label = SECTION_LABELS.get(section_key, section_key)
        google_html += f'<div class="subsection-label">{label}</div>'
        for i, item in enumerate(items[:3]):
            local_path   = save_article_file(item, i+1, f'{category}_{section_key}', output_dir)
            google_html += build_news_item_html(item, local_path, 'google')
            google_count += 1

    if not google_html:
        google_html = '<div class="empty-msg">추가 뉴스가 없습니다.</div>'

    # ── 연구 동향 ─────────────────────────────────────────────
    research_items = parse_research_from_report(report_text)
    research_html  = ''
    for i, item in enumerate(research_items):
        local_path    = save_paper_file(item, i+1, category, output_dir)
        research_html += build_paper_item_html(item, local_path)
    if not research_html:
        research_html = '<div class="empty-msg">수집된 논문이 없습니다.</div>'

    # ── 실무 적용 포인트 ──────────────────────────────────────
    actions      = parse_actions_only(report_text)
    actions_html = ''
    for i, action in enumerate(actions, 1):
        actions_html += f'''
        <li class="action-item">
            <span class="action-num">0{i}</span>
            <div class="action-text">{action}</div>
        </li>'''
    if not actions_html:
        actions_html = '<li class="action-item"><span class="action-num">-</span><div class="action-text">실무 포인트가 없습니다.</div></li>'

    return f'''
<div class="tab-content {'active' if tab_num == 1 else ''}" id="tab-{category}">

  <!-- 핵심 요약 -->
  <div class="summary-card" data-color="{color_var}">
    <div class="summary-label">🔑 이번 주 핵심 요약</div>
    <ul class="summary-items">{summary_html}</ul>
  </div>

  <!-- 2×2 그리드 -->
  <div class="sections-grid">

    <!-- 국내 주요 뉴스 -->
    <div class="section-card">
      <div class="section-title">📰 국내 주요 뉴스</div>
      <div class="section-desc">AI타임스 · ZDNet Korea · IT조선 · GeekNews</div>
      {domestic_html}
    </div>

    <!-- 글로벌 동향 -->
    <div class="section-card">
      <div class="section-title">🌐 글로벌 동향</div>
      <div class="section-desc">TechCrunch · The Verge (한국어 요약)</div>
      {global_html}
    </div>

    <!-- 추가 뉴스 -->
    <div class="section-card">
      <div class="section-title">📡 추가 뉴스</div>
      <div class="section-desc">Google News 키워드 수집</div>
      {google_html}
    </div>

    <!-- 연구 동향 -->
    <div class="section-card">
      <div class="section-title">📚 연구 동향</div>
      <div class="section-desc">arXiv 논문 핵심 요약</div>
      {research_html}
    </div>

  </div>

  <!-- 실무 적용 포인트 (전체 너비) -->
  <div class="action-card">
    <div class="section-title">💡 실무 적용 포인트</div>
    <ul class="action-items">{actions_html}</ul>
  </div>

</div>'''


# ── HTML 전체 생성 ──────────────────────────────────────────────

def generate_html(reports: dict, week_label: str, output_dir: str) -> str:
    now        = datetime.now(KST)
    date_range = f"{now.year}.{now.month:02d}"

    categories = [
        ('AI',            'ai'),
        ('데이터엔지니어링', 'de'),
        ('RPA',           'rpa'),
    ]

    tab_buttons  = ''
    tab_contents = ''

    for i, (cat, color) in enumerate(categories, 1):
        active = 'active' if i == 1 else ''
        tab_buttons += f'''
    <button class="tab-btn {active}" data-tab="{cat}" onclick="switchTab('{cat}')">
        <span class="tab-dot"></span>{cat}
    </button>'''

        data   = reports.get(cat, {'report_text': '', 'sections_json': {}})
        result = build_tab_html(cat, color, data, i, output_dir)
        if result:
            tab_contents += result

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IT 주간 동향 | {week_label}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Bebas+Neue&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {{
    --bg:#0a0a0f; --surface:#13131a; --surface2:#1c1c28;
    --border:#2a2a3a; --text:#e8e8f0; --muted:#7878a0;
    --ai:#6c63ff; --ai-dim:#6c63ff22;
    --de:#00d4aa; --de-dim:#00d4aa22;
    --rpa:#ff6b6b; --rpa-dim:#ff6b6b22;
    --accent:#f0c040;
    --domestic:#4a9eff;
    --global:#ff9f40;
    --google:#66bb6a;
    --arxiv:#a78bfa;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Noto Sans KR',sans-serif; background:var(--bg);
    color:var(--text); min-height:100vh; font-size:14px; line-height:1.7; }}

  /* 헤더 */
  header {{ padding:40px 60px 30px; border-bottom:1px solid var(--border);
    display:flex; justify-content:space-between; align-items:flex-end;
    position:relative; overflow:hidden; }}
  header::before {{ content:''; position:absolute; top:-60px; left:-60px;
    width:300px; height:300px;
    background:radial-gradient(circle,#6c63ff33 0%,transparent 70%);
    pointer-events:none; }}
  .header-title {{ font-family:'Bebas Neue',sans-serif; font-size:48px;
    letter-spacing:4px; color:#fff; line-height:1; }}
  .header-sub {{ font-size:13px; color:var(--muted); margin-top:6px;
    font-family:'JetBrains Mono',monospace; letter-spacing:2px; }}
  .header-badge {{ display:inline-block; background:var(--accent); color:#000;
    font-size:11px; font-weight:700; padding:4px 12px;
    letter-spacing:2px; font-family:'JetBrains Mono',monospace; }}
  .header-date {{ display:block; font-size:12px; color:var(--muted);
    margin-top:6px; font-family:'JetBrains Mono',monospace; }}

  /* 탭 */
  .tab-nav {{ display:flex; padding:0 60px; border-bottom:1px solid var(--border);
    position:sticky; top:0; background:var(--bg); z-index:100; }}
  .tab-btn {{ padding:16px 32px; border:none; background:none; color:var(--muted);
    font-family:'Noto Sans KR',sans-serif; font-size:13px; font-weight:500;
    cursor:pointer; position:relative; transition:color 0.2s; letter-spacing:1px; }}
  .tab-btn::after {{ content:''; position:absolute; bottom:-1px; left:0;
    right:0; height:2px; background:transparent; transition:background 0.2s; }}
  .tab-btn.active[data-tab="AI"] {{ color:var(--ai); }}
  .tab-btn.active[data-tab="AI"]::after {{ background:var(--ai); }}
  .tab-btn.active[data-tab="데이터엔지니어링"] {{ color:var(--de); }}
  .tab-btn.active[data-tab="데이터엔지니어링"]::after {{ background:var(--de); }}
  .tab-btn.active[data-tab="RPA"] {{ color:var(--rpa); }}
  .tab-btn.active[data-tab="RPA"]::after {{ background:var(--rpa); }}
  .tab-dot {{ display:inline-block; width:8px; height:8px;
    border-radius:50%; margin-right:8px; }}
  .tab-btn[data-tab="AI"] .tab-dot {{ background:var(--ai); }}
  .tab-btn[data-tab="데이터엔지니어링"] .tab-dot {{ background:var(--de); }}
  .tab-btn[data-tab="RPA"] .tab-dot {{ background:var(--rpa); }}

  /* 콘텐츠 */
  .tab-content {{ display:none; padding:40px 60px 60px; }}
  .tab-content.active {{ display:block; }}

  /* 핵심 요약 */
  .summary-card {{ border-radius:2px; padding:24px 28px; margin-bottom:28px; }}
  .summary-card[data-color="ai"]  {{ background:var(--ai-dim);  border-left:3px solid var(--ai); }}
  .summary-card[data-color="de"]  {{ background:var(--de-dim);  border-left:3px solid var(--de); }}
  .summary-card[data-color="rpa"] {{ background:var(--rpa-dim); border-left:3px solid var(--rpa); }}
  .summary-label {{ font-family:'JetBrains Mono',monospace; font-size:10px;
    letter-spacing:3px; text-transform:uppercase; margin-bottom:12px; opacity:0.7; }}
  .summary-items {{ list-style:none; }}
  .summary-items li {{ padding:5px 0; padding-left:14px; position:relative;
    font-size:14px; line-height:1.6; }}
  .summary-items li::before {{ content:'›'; position:absolute; left:0;
    font-size:16px; opacity:0.6; }}

  /* 2×2 그리드 */
  .sections-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px;
    margin-bottom:16px; }}
  .section-card {{ background:var(--surface); border:1px solid var(--border);
    border-radius:2px; padding:20px; }}
  .section-card:hover {{ border-color:#3a3a55; }}
  .section-title {{ font-size:13px; font-weight:700; letter-spacing:1px;
    font-family:'JetBrains Mono',monospace; margin-bottom:4px; }}
  .section-desc {{ font-size:11px; color:var(--muted); margin-bottom:14px;
    font-family:'JetBrains Mono',monospace; }}
  .subsection-label {{ font-size:11px; color:var(--muted); font-weight:700;
    letter-spacing:1px; padding:8px 0 4px;
    border-top:1px solid var(--border); margin-top:8px; }}
  .subsection-label:first-child {{ border-top:none; margin-top:0; }}

  /* 뉴스 아이템 */
  .news-item {{ padding:12px 0; border-bottom:1px solid var(--border); }}
  .news-item:last-of-type {{ border-bottom:none; }}
  .news-item-header {{ display:flex; align-items:flex-start;
    justify-content:space-between; gap:8px; margin-bottom:6px; }}
  .news-title {{ font-size:13px; font-weight:500; color:var(--text);
    line-height:1.5; flex:1; }}
  .news-summary {{ font-size:12px; color:var(--muted); line-height:1.6;
    margin-bottom:6px; }}
  .paper-insight {{ font-size:12px; color:var(--arxiv); line-height:1.5;
    margin-bottom:6px; padding:6px 10px;
    background:rgba(167,139,250,0.08); border-radius:2px; }}
  .news-footer {{ display:flex; align-items:center;
    justify-content:space-between; gap:8px; }}
  .news-url {{ font-size:10px; color:#3a3a5a;
    font-family:'JetBrains Mono',monospace;
    word-break:break-all; flex:1; }}
  .detail-link {{ flex-shrink:0; font-size:11px;
    font-family:'JetBrains Mono',monospace;
    color:var(--muted); text-decoration:none;
    border:1px solid var(--border); padding:2px 8px;
    border-radius:2px; transition:all 0.15s; white-space:nowrap; }}
  .detail-link:hover {{ color:var(--text); border-color:#4a4a6a;
    background:var(--surface2); }}
  .arxiv-link {{ color:var(--arxiv); border-color:rgba(167,139,250,0.3); }}
  .arxiv-link:hover {{ background:rgba(167,139,250,0.1); }}

  /* 소스 뱃지 */
  .source-badge {{ flex-shrink:0; font-size:10px; padding:2px 7px;
    border-radius:2px; font-family:'JetBrains Mono',monospace;
    letter-spacing:1px; white-space:nowrap; }}
  .source-badge.domestic {{ background:rgba(74,158,255,0.15);   color:var(--domestic); }}
  .source-badge.global   {{ background:rgba(255,159,64,0.15);   color:var(--global); }}
  .source-badge.google   {{ background:rgba(102,187,106,0.15);  color:var(--google); }}
  .source-badge.arxiv    {{ background:rgba(167,139,250,0.15);  color:var(--arxiv); }}

  .empty-msg {{ font-size:12px; color:var(--muted); padding:12px 0; }}

  /* 실무 포인트 */
  .action-card {{ background:var(--surface2); border:1px solid var(--border);
    border-radius:2px; padding:20px; margin-top:0; }}
  .action-items {{ list-style:none; }}
  .action-item {{ display:flex; gap:14px; padding:12px 0;
    border-bottom:1px solid var(--border); align-items:flex-start; }}
  .action-item:last-child {{ border-bottom:none; padding-bottom:0; }}
  .action-num {{ font-family:'Bebas Neue',sans-serif; font-size:28px;
    line-height:1; opacity:0.15; flex-shrink:0; min-width:26px; }}
  .action-text {{ font-size:13px; line-height:1.6; padding-top:2px;
    color:var(--muted); }}

  /* 푸터 */
  footer {{ padding:20px 60px; border-top:1px solid var(--border);
    display:flex; justify-content:space-between; align-items:center; }}
  footer span {{ font-family:'JetBrains Mono',monospace; font-size:11px;
    color:var(--muted); letter-spacing:1px; }}
</style>
</head>
<body>

<header>
  <div>
    <div class="header-title">IT WEEKLY<br>TREND REPORT</div>
    <div class="header-sub">AI · DATA ENGINEERING · RPA</div>
  </div>
  <div style="text-align:right">
    <span class="header-badge">{week_label.upper()}</span>
    <span class="header-date">{date_range}</span>
  </div>
</header>

<nav class="tab-nav">
  {tab_buttons}
</nav>

{tab_contents}

<footer>
  <span>IT WEEKLY TREND REPORT — {week_label}</span>
  <span>AUTO-GENERATED · TREND-BOX SYSTEM</span>
</footer>

<script>
function switchTab(tab) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`[data-tab="${{tab}}"]`).classList.add('active');
  document.getElementById(`tab-${{tab}}`).classList.add('active');
}}
</script>
</body>
</html>'''


# ── ZIP 패키징 ──────────────────────────────────────────────────

def create_zip(output_dir: str, week_label: str) -> str:
    """output 폴더를 ZIP으로 압축"""
    safe_label = week_label.replace(' ', '_').replace('년', '').replace('월', '').replace('주차', 'w')
    zip_name   = f"trend_report_{safe_label}.zip"
    zip_path   = os.path.join(os.path.dirname(output_dir), zip_name)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname  = os.path.relpath(filepath, os.path.dirname(output_dir))
                zf.write(filepath, arcname)

    size_kb = os.path.getsize(zip_path) // 1024
    print(f"✅ ZIP 생성 완료: {zip_path} ({size_kb}KB)")
    return zip_path


# ── 메인 ────────────────────────────────────────────────────────

def main():
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if not all([supabase_url, supabase_key]):
        raise ValueError("환경변수 누락: SUPABASE_URL, SUPABASE_KEY 확인")

    supabase   = create_client(supabase_url, supabase_key)
    week_label = get_week_label()
    print(f"리포트 생성: {week_label}")

    reports = fetch_reports(supabase, week_label)
    if not reports:
        print(f"❌ {week_label} 리포트가 DB에 없습니다.")
        return

    print(f"✅ {list(reports.keys())} 카테고리 로드 완료")

    # 출력 폴더 설정
    safe_label  = week_label.replace(' ', '_').replace('년', '').replace('월', '').replace('주차', 'w')
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    output_dir  = os.path.join(base_dir, f"output_{safe_label}")
    os.makedirs(output_dir, exist_ok=True)

    # HTML 생성 (articles/, papers/ 폴더도 함께 생성됨)
    html = generate_html(reports, week_label, output_dir)

    # 메인 리포트 저장
    report_path = os.path.join(output_dir, 'weekly_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ 리포트 HTML 저장: {report_path}")

    # ZIP 패키징
    zip_path = create_zip(output_dir, week_label)

    print(f"\n📦 배포 방법:")
    print(f"   {zip_path} 를 내부망으로 전달")
    print(f"   압축 해제 후 weekly_report.html 열기")
    print(f"   모든 링크가 로컬에서 작동합니다")


if __name__ == "__main__":
    main()