# ============================================================
# FRED API 퀀트 3콤보 분석 대시보드 (Plotly Interactive Version)
# Net Liquidity / Dollar Index / HY Spread vs BTC/NASDAQ/S&P500
# ============================================================

# 1. 필요한 라이브러리 설치 및 임포트
!pip install fredapi plotly kaleido -q

from fredapi import Fred
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 2. FRED API 설정
# ============================================================
FRED_API_KEY = "f1137018f7bb4b4150a5c84e09fc7fc2"
fred = Fred(api_key=FRED_API_KEY)

print("🔄 FRED 데이터 다운로드 시작...")

# ============================================================
# 3. 데이터 수집
# ============================================================

try:
    # --- 3-1. Net Liquidity 구성 요소 ---
    print("📥 Net Liquidity 데이터 다운로드 중...")
    walcl = fred.get_series('WALCL')        # Fed Total Assets (Millions)
    tga = fred.get_series('WTREGEN')        # Treasury General Account (Millions)
    rrp = fred.get_series('RRPONTSYD')      # Reverse Repo (Billions)

    # --- 3-2. 달러 인덱스 ---
    print("📥 Dollar Index 데이터 다운로드 중...")
    dxy = fred.get_series('DTWEXAFEGS')     # Dollar Index (Advanced Foreign Economies)

    # --- 3-3. High Yield Spread ---
    print("📥 High Yield Spread 데이터 다운로드 중...")
    hy_spread = fred.get_series('BAMLH0A0HYM2')  # ICE BofA US HY Spread

    # --- 3-4. 자산 가격 ---
    print("📥 자산 가격 데이터 다운로드 중...")
    btc = fred.get_series('CBBTCUSD')       # Bitcoin (Coinbase)
    nasdaq = fred.get_series('NASDAQCOM')   # NASDAQ Composite
    sp500 = fred.get_series('SP500')        # S&P 500

    print("✅ 모든 데이터 다운로드 완료!")

except Exception as e:
    print(f"❌ 데이터 다운로드 오류: {e}")
    raise

# ============================================================
# 4. Net Liquidity 계산
# ============================================================

df_liq = pd.concat([walcl, tga, rrp], axis=1)
df_liq.columns = ['WALCL_Mn', 'TGA_Mn', 'RRP_Bn']

# 단위 통일: RRP (Billions -> Millions)
df_liq['RRP_Mn'] = df_liq['RRP_Bn'] * 1000

# 결측치 처리 (forward fill)
df_liq = df_liq.sort_index().ffill().dropna()

# Net Liquidity 계산
df_liq['NetLiquidity'] = (
    df_liq['WALCL_Mn'] - df_liq['TGA_Mn'] - df_liq['RRP_Mn']
)

print(f"📊 Net Liquidity 계산 완료: {len(df_liq)} 데이터 포인트")

# ============================================================
# 5. 전체 데이터 통합
# ============================================================

df_all = pd.concat([
    df_liq['NetLiquidity'],
    dxy,
    hy_spread,
    btc,
    nasdaq,
    sp500
], axis=1, join='inner').dropna()

df_all.columns = ['NetLiq', 'DXY', 'HYSpread', 'BTC', 'NASDAQ', 'SP500']

# 최근 3년 데이터만 사용
start_date = datetime.now() - timedelta(days=3*365)
df_recent = df_all[df_all.index >= start_date].copy()

print(f"📅 분석 기간: {df_recent.index[0].date()} ~ {df_recent.index[-1].date()}")
print(f"📈 데이터 포인트: {len(df_recent)}개")

# ============================================================
# 6. 정규화 함수
# ============================================================

def zscore(series):
    """Z-score 정규화"""
    return (series - series.mean()) / series.std()

# ============================================================
# 7. Plotly 시각화 1: 콤보 1 - Net Liquidity vs BTC/NASDAQ
# ============================================================

# Z-score 정규화
df_z1 = df_recent[['NetLiq', 'BTC', 'NASDAQ']].apply(zscore)

# 롤링 상관계수 계산
ret = df_recent[['NetLiq', 'BTC', 'NASDAQ']].pct_change().dropna()
window = 90
corr_btc = ret['NetLiq'].rolling(window).corr(ret['BTC'])
corr_nasdaq = ret['NetLiq'].rolling(window).corr(ret['NASDAQ'])

# Net Liquidity 변화율
netliq_change = df_recent['NetLiq'].pct_change(periods=60) * 100

# 서브플롯 생성
fig1 = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        '콤보 1: Net Liquidity vs BTC/NASDAQ (Z-score)',
        f'Net Liquidity 상관계수 ({window}일 롤링)',
        'Net Liquidity 60일 변화율 (유동성 확장/축소)'
    ),
    vertical_spacing=0.08,
    row_heights=[0.35, 0.3, 0.35]
)

# 7-1. Z-score 오버레이
fig1.add_trace(
    go.Scatter(x=df_z1.index, y=df_z1['NetLiq'],
               name='Net Liquidity', line=dict(color='#2E86AB', width=2.5)),
    row=1, col=1
)
fig1.add_trace(
    go.Scatter(x=df_z1.index, y=df_z1['BTC'],
               name='Bitcoin', line=dict(color='#F77F00', width=2.5)),
    row=1, col=1
)
fig1.add_trace(
    go.Scatter(x=df_z1.index, y=df_z1['NASDAQ'],
               name='NASDAQ', line=dict(color='#06A77D', width=2.5)),
    row=1, col=1
)
fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

# 7-2. 롤링 상관계수
fig1.add_trace(
    go.Scatter(x=corr_btc.index, y=corr_btc,
               name='Corr(NetLiq, BTC)',
               line=dict(color='#F77F00', width=2.5),
               fill='tozeroy', fillcolor='rgba(247, 127, 0, 0.2)'),
    row=2, col=1
)
fig1.add_trace(
    go.Scatter(x=corr_nasdaq.index, y=corr_nasdaq,
               name='Corr(NetLiq, NASDAQ)',
               line=dict(color='#06A77D', width=2.5),
               fill='tozeroy', fillcolor='rgba(6, 167, 125, 0.2)'),
    row=2, col=1
)
fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

# 7-3. Net Liquidity 변화율
# 확장 구간
expansion = netliq_change[netliq_change > 0]
fig1.add_trace(
    go.Scatter(x=expansion.index, y=expansion,
               name='확장 구간 🟢',
               line=dict(color='#06A77D', width=0),
               fill='tozeroy', fillcolor='rgba(6, 167, 125, 0.4)'),
    row=3, col=1
)

# 축소 구간
contraction = netliq_change[netliq_change <= 0]
fig1.add_trace(
    go.Scatter(x=contraction.index, y=contraction,
               name='축소 구간 🔴',
               line=dict(color='#D62828', width=0),
               fill='tozeroy', fillcolor='rgba(214, 40, 40, 0.4)'),
    row=3, col=1
)

# 변화율 라인
fig1.add_trace(
    go.Scatter(x=netliq_change.index, y=netliq_change,
               name='변화율', line=dict(color='black', width=2),
               showlegend=False),
    row=3, col=1
)
fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

# 레이아웃 설정
fig1.update_layout(
    height=1200,
    title_text="<b>퀀트 콤보 1: Net Liquidity 분석</b>",
    title_font_size=20,
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

fig1.update_yaxes(title_text="Z-score", row=1, col=1)
fig1.update_yaxes(title_text="Correlation", row=2, col=1, range=[-1, 1])
fig1.update_yaxes(title_text="변화율 (%)", row=3, col=1)

fig1.show()
fig1.write_html('combo1_net_liquidity.html')

# ============================================================
# 8. Plotly 시각화 2: 콤보 2 - Dollar Index vs BTC
# ============================================================

# DXY 반전 vs BTC
df_z2 = pd.DataFrame({
    'DXY_Inverted': zscore(-df_recent['DXY']),
    'BTC': zscore(df_recent['BTC'])
})

# 롤링 상관계수
ret2 = df_recent[['DXY', 'BTC']].pct_change().dropna()
corr_dxy_btc = ret2['DXY'].rolling(window).corr(ret2['BTC'])

fig2 = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        '콤보 2: Dollar Index (반전) vs BTC (Z-score)',
        f'Dollar Index vs BTC 상관계수 ({window}일 롤링)'
    ),
    vertical_spacing=0.12,
    row_heights=[0.5, 0.5]
)

# 8-1. DXY 반전 vs BTC
fig2.add_trace(
    go.Scatter(x=df_z2.index, y=df_z2['DXY_Inverted'],
               name='Dollar Index (반전)',
               line=dict(color='#D62828', width=2.5)),
    row=1, col=1
)
fig2.add_trace(
    go.Scatter(x=df_z2.index, y=df_z2['BTC'],
               name='Bitcoin',
               line=dict(color='#F77F00', width=2.5)),
    row=1, col=1
)
fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

# 8-2. 롤링 상관계수
fig2.add_trace(
    go.Scatter(x=corr_dxy_btc.index, y=corr_dxy_btc,
               name='Correlation',
               line=dict(color='#9D4EDD', width=2.5),
               fill='tozeroy', fillcolor='rgba(157, 78, 221, 0.3)'),
    row=2, col=1
)
fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

fig2.update_layout(
    height=900,
    title_text="<b>퀀트 콤보 2: Dollar Index vs Bitcoin 분석</b>",
    title_font_size=20,
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

fig2.update_yaxes(title_text="Z-score", row=1, col=1)
fig2.update_yaxes(title_text="Correlation", row=2, col=1, range=[-1, 1])

fig2.show()
fig2.write_html('combo2_dollar_btc.html')

# ============================================================
# 9. Plotly 시각화 3: 콤보 3 - HY Spread vs S&P 500
# ============================================================

# 롤링 상관계수
ret3 = df_recent[['HYSpread', 'SP500']].pct_change().dropna()
corr_hy_sp = ret3['HYSpread'].rolling(window).corr(ret3['SP500'])

# Divergence 감지
sp_ret = df_recent['SP500'].pct_change(periods=20)
hy_change = df_recent['HYSpread'].diff(periods=20)
divergence = (sp_ret > 0) & (hy_change > 0)

fig3 = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        '콤보 3: High Yield Spread vs S&P 500',
        f'HY Spread vs S&P 500 상관계수 ({window}일 롤링)',
        'Divergence 감지: S&P 상승 + HY Spread 상승 (매도 신호)'
    ),
    specs=[[{"secondary_y": True}],
           [{"secondary_y": False}],
           [{"secondary_y": False}]],
    vertical_spacing=0.08,
    row_heights=[0.35, 0.3, 0.35]
)

# 9-1. HY Spread vs S&P 500 (이중 축)
fig3.add_trace(
    go.Scatter(x=df_recent.index, y=df_recent['SP500'],
               name='S&P 500',
               line=dict(color='#2E86AB', width=2.5)),
    row=1, col=1, secondary_y=False
)
fig3.add_trace(
    go.Scatter(x=df_recent.index, y=df_recent['HYSpread'],
               name='HY Spread',
               line=dict(color='#D62828', width=2.5)),
    row=1, col=1, secondary_y=True
)

# 위험 구간 표시
danger_zone = df_recent[df_recent['HYSpread'] > 5.0]
if len(danger_zone) > 0:
    fig3.add_hline(y=5.0, line_dash="dash", line_color="darkred",
                   line_width=2.5, opacity=0.8,
                   annotation_text="위기 임계점 (5%)",
                   annotation_position="right",
                   row=1, col=1, secondary_y=True)
    fig3.add_trace(
        go.Scatter(x=danger_zone.index, y=danger_zone['HYSpread'],
                   fill='tonexty', fillcolor='rgba(214, 40, 40, 0.3)',
                   line=dict(width=0), showlegend=False),
        row=1, col=1, secondary_y=True
    )

# 9-2. 롤링 상관계수
fig3.add_trace(
    go.Scatter(x=corr_hy_sp.index, y=corr_hy_sp,
               name='Correlation',
               line=dict(color='#A4133C', width=2.5),
               fill='tozeroy', fillcolor='rgba(164, 19, 60, 0.3)'),
    row=2, col=1
)
fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

# 9-3. Divergence 감지
fig3.add_trace(
    go.Scatter(x=df_recent.index, y=df_recent['SP500'],
               name='S&P 500',
               line=dict(color='#2E86AB', width=2), opacity=0.6),
    row=3, col=1
)
fig3.add_trace(
    go.Scatter(x=df_recent[divergence].index,
               y=df_recent.loc[divergence, 'SP500'],
               name='Divergence 경고 ⚠️',
               mode='markers',
               marker=dict(color='red', size=10, symbol='diamond')),
    row=3, col=1
)

fig3.update_layout(
    height=1200,
    title_text="<b>퀀트 콤보 3: High Yield Spread 분석</b>",
    title_font_size=20,
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

fig3.update_yaxes(title_text="S&P 500", row=1, col=1, secondary_y=False)
fig3.update_yaxes(title_text="HY Spread (%)", row=1, col=1, secondary_y=True)
fig3.update_yaxes(title_text="Correlation", row=2, col=1, range=[-1, 1])
fig3.update_yaxes(title_text="S&P 500", row=3, col=1)

fig3.show()
fig3.write_html('combo3_hy_spread.html')

# ============================================================
# 10. 종합 대시보드
# ============================================================

# 상관계수 매트릭스
corr_matrix = df_recent[['NetLiq', 'DXY', 'HYSpread', 'BTC', 'NASDAQ', 'SP500']].corr()

# 최신 지표
latest = df_recent.iloc[-1]
netliq_60d = df_recent['NetLiq'].pct_change(periods=60).iloc[-1] * 100

fig_dashboard = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Net Liquidity + BTC/NASDAQ (Z-score)',
        '상관계수 히트맵',
        'Dollar Index (반전) vs BTC',
        'High Yield Spread',
        '최신 지표 요약',
        '롤링 상관계수 종합'
    ),
    specs=[
        [{"colspan": 2}, None],
        [{"type": "xy"}, {"type": "heatmap"}],
        [{"type": "table"}, {"type": "xy"}]
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.15,
    row_heights=[0.3, 0.35, 0.35]
)

# 10-1. Net Liquidity + BTC/NASDAQ
df_z_all = df_recent[['NetLiq', 'BTC', 'NASDAQ']].apply(zscore)
fig_dashboard.add_trace(
    go.Scatter(x=df_z_all.index, y=df_z_all['NetLiq'],
               name='Net Liquidity', line=dict(color='#2E86AB', width=2.5)),
    row=1, col=1
)
fig_dashboard.add_trace(
    go.Scatter(x=df_z_all.index, y=df_z_all['BTC'],
               name='Bitcoin', line=dict(color='#F77F00', width=2.5)),
    row=1, col=1
)
fig_dashboard.add_trace(
    go.Scatter(x=df_z_all.index, y=df_z_all['NASDAQ'],
               name='NASDAQ', line=dict(color='#06A77D', width=2.5)),
    row=1, col=1
)
fig_dashboard.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

# 10-2. Dollar Index (반전) vs BTC
fig_dashboard.add_trace(
    go.Scatter(x=df_z2.index, y=df_z2['DXY_Inverted'],
               name='DXY (반전)', line=dict(color='#D62828', width=2)),
    row=2, col=1
)
fig_dashboard.add_trace(
    go.Scatter(x=df_z2.index, y=df_z2['BTC'],
               name='BTC', line=dict(color='#F77F00', width=2)),
    row=2, col=1
)
fig_dashboard.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

# 10-3. 상관계수 히트맵
fig_dashboard.add_trace(
    go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ),
    row=2, col=2
)

# 10-4. HY Spread
fig_dashboard.add_trace(
    go.Scatter(x=df_recent.index, y=df_recent['HYSpread'],
               name='HY Spread', line=dict(color='#A4133C', width=2)),
    row=3, col=2
)
fig_dashboard.add_hline(y=5.0, line_dash="dash", line_color="red",
                       line_width=2, opacity=0.8, row=3, col=2)

# 10-5. 최신 지표 요약 테이블
summary_data = [
    ['지표', '현재값', '신호'],
    ['Net Liquidity', f'${latest["NetLiq"]/1e6:.2f}T',
     f'{"🟢 확장" if netliq_60d > 0 else "🔴 축소"} ({netliq_60d:+.1f}%)'],
    ['Dollar Index', f'{latest["DXY"]:.2f}', ''],
    ['HY Spread', f'{latest["HYSpread"]:.2f}%',
     f'{"⚠️ 위험" if latest["HYSpread"] > 5 else "✅ 정상"}'],
    ['Bitcoin', f'${latest["BTC"]:,.0f}', ''],
    ['NASDAQ', f'{latest["NASDAQ"]:,.0f}', ''],
    ['S&P 500', f'{latest["SP500"]:.2f}', ''],
    ['NetLiq ↔ BTC 상관', f'{corr_btc.iloc[-1]:.3f}', ''],
    ['DXY ↔ BTC 상관', f'{corr_dxy_btc.iloc[-1]:.3f}', '역상관'],
    ['HY ↔ SP500 상관', f'{corr_hy_sp.iloc[-1]:.3f}', '']
]

fig_dashboard.add_trace(
    go.Table(
        header=dict(values=['<b>지표</b>', '<b>현재값</b>', '<b>신호</b>'],
                   fill_color='#2E86AB',
                   font=dict(color='white', size=12),
                   align='left'),
        cells=dict(values=list(zip(*summary_data)),
                  fill_color='lavender',
                  font=dict(size=11),
                  align='left',
                  height=25)
    ),
    row=3, col=1
)

fig_dashboard.update_layout(
    height=1400,
    title_text=f"<b>퀀트 3콤보 종합 대시보드</b><br><sub>분석 기간: {df_recent.index[0].date()} ~ {df_recent.index[-1].date()}</sub>",
    title_font_size=22,
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

fig_dashboard.show()
fig_dashboard.write_html('dashboard_comprehensive.html')

# ============================================================
# 11. 수치 요약 출력
# ============================================================

print("\n" + "="*70)
print("📊 퀀트 3콤보 분석 완료!")
print("="*70)
print(f"\n📅 분석 기간: {df_recent.index[0].date()} ~ {df_recent.index[-1].date()}")
print(f"\n💰 현재 Net Liquidity: ${latest['NetLiq']/1e6:.2f} Trillion")
print(f"   └ 60일 변화율: {netliq_60d:+.2f}%")
print(f"   └ 시그널: {'확장 (리스크 온) 🟢' if netliq_60d > 0 else '축소 (리스크 오프) 🔴'}")
print(f"\n💵 Dollar Index: {latest['DXY']:.2f}")
print(f"\n⚠️ HY Spread: {latest['HYSpread']:.2f}% {'(위험 구간 🔴)' if latest['HYSpread'] > 5 else '(정상 구간 🟢)'}")
print(f"\n🔗 최근 {window}일 상관계수:")
print(f"   • Net Liq ↔ BTC: {corr_btc.iloc[-1]:+.3f}")
print(f"   • Net Liq ↔ NASDAQ: {corr_nasdaq.iloc[-1]:+.3f}")
print(f"   • DXY ↔ BTC: {corr_dxy_btc.iloc[-1]:+.3f} (역상관 강도)")
print(f"   • HY Spread ↔ SP500: {corr_hy_sp.iloc[-1]:+.3f}")
print("\n✅ 인터랙티브 HTML 그래프 4개 생성 완료:")
print("   1. combo1_net_liquidity.html")
print("   2. combo2_dollar_btc.html")
print("   3. combo3_hy_spread.html")
print("   4. dashboard_comprehensive.html")
print("\n💡 파일을 다운로드하여 브라우저에서 열면 줌/팬/호버 기능 사용 가능!")
print("="*70)

# ============================================================
# 12. 보너스: 간단한 트레이딩 시그널 생성
# ============================================================

print("\n" + "="*70)
print("🎯 현재 트레이딩 시그널")
print("="*70)

# 시그널 1: Net Liquidity 확장/축소
if netliq_60d > 2:
    print("\n✅ 시그널 1: Net Liquidity 강한 확장 (+2% 이상)")
    print("   → BTC/NASDAQ 매수 고려")
elif netliq_60d < -2:
    print("\n⚠️ 시그널 1: Net Liquidity 강한 축소 (-2% 이하)")
    print("   → 리스크 자산 매도/경계")
else:
    print("\n⏸️ 시그널 1: Net Liquidity 중립 구간")

# 시그널 2: DXY vs BTC 역상관
if corr_dxy_btc.iloc[-1] < -0.5:
    print("\n✅ 시그널 2: DXY-BTC 강한 역상관 (-0.5 이하)")
    print("   → DXY 하락 시 BTC 매수 기회")
else:
    print("\n⏸️ 시그널 2: DXY-BTC 역상관 약화")

# 시그널 3: HY Spread 위험 신호
if latest['HYSpread'] > 5.0:
    print("\n🚨 시그널 3: HY Spread 위기 임계점 초과 (5% 이상)")
    print("   → 주식 시장 위험! 매도/방어 전략")
elif latest['HYSpread'] > 4.0:
    print("\n⚠️ 시그널 3: HY Spread 경계 구간 (4-5%)")
    print("   → 주의 필요, 포지션 축소 고려")
else:
    print("\n✅ 시그널 3: HY Spread 정상 구간")
    print("   → 주식 시장 안정")

# Divergence 체크
recent_divergence = divergence.tail(5).sum()
if recent_divergence > 0:
    print(f"\n⚠️ 경고: 최근 5일 중 {recent_divergence}일 Divergence 발생!")
    print("   → S&P 상승 + HY Spread 상승 = 매도 신호")

print("="*70)