# 청년 고용 데이터 시각화 (발표자료용 막대그래프 2장)
import os
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 깨짐 방지 (윈도우 맑은 고딕)
os.makedirs("results", exist_ok=True)           # results 폴더가 없으면 생성

def draw(labels, values, title, source, filename, horizontal=False):
    # 가장 큰 값만 진한 보라, 나머지는 연보라
    colors = ["#4b1f6f" if v == max(values) else "#c9c2d8" for v in values]
    fig, ax = plt.subplots(figsize=(8, 4.5))    # 16:9 비율
    if horizontal:
        ax.barh(labels[::-1], values[::-1], color=colors[::-1])  # 첫 항목이 맨 위로
    else:
        ax.bar(labels, values, color=colors)
    ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=12)    # 막대에 숫자
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)              # 위·오른쪽 테두리 제거
    fig.text(0.01, 0.01, "출처: " + source, fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig("results/" + filename, dpi=200)
    plt.close()

# 그래프 1
draw(["2022", "2024", "2025", "2026"], [27.7, 30.8, 34.1, 35.2],
     '쉬는 청년 3명 중 1명 이상, "원하는 일자리를 찾기 어려워서"',
     "국가데이터처 비경제활동인구 부가조사 (15~29세, 각 연도 8월)",
     "청년_원하는일자리.png")

# 그래프 2
draw(["직업훈련·교육", "취업알선·정보 제공", "생활비 지원"], [59.3, 54.7, 50.6],
     '쉬었음 청년이 원하는 도움: 돈보다 "일을 알 기회"',
     "한국고용정보원 쉬었음 청년 실태조사 (2025.3)",
     "청년_원하는도움.png", horizontal=True)

print("완료!")
