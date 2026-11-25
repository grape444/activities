# ============================================================
# analyze.py : 본선 AI 채점 신뢰도 검증
# ============================================================
# [입력]: AI 채점 기록 105개(21답안 × 5회) + 내 기대 판정 21개
# [순서]
# ① 불러오기·표 
# ② 정확성 
# ③ 일관성 
# ④ 일치율
# ⑤ 특수 검사 
# ⑥ 불일치 분석 
# ⑦ 혼동행렬·카파 
# ⑧ 그래프
# ============================================================
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, cohen_kappa_score

plt.rcParams["font.family"] = ["Malgun Gothic", "Noto Sans CJK KR"]  # 한글 폰트 (윈도우 / 리눅스)
plt.rcParams["axes.unicode_minus"] = False                          # 마이너스 기호 깨짐 방지

LEVEL = {"미충족": 0, "부분": 1, "충족": 2}   # 판정을 숫자로 (가중 카파용, 순서가 있음)
LABELS = ["충족", "부분", "미충족"]          # 혼동행렬 행·열 순서


# ① 불러오기 + 표 만들기 ------------------------------------------
def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_table(result, expected):
    # AI 채점 기록: 채점 1회 = 1행
    df = pd.DataFrame([
        {"answer_id": r["answer_id"], "run": r["run"], "status": r["status"],
         "total": r["total"], "penalties": sorted(r["penalties"]), "error": r["error"]}
        for r in result["records"]
    ])
    # 정답표: 답안 1개 = 1행
    exp = pd.DataFrame([
        {"answer_id": a["answer_id"], "job": a["job"], "type": a["type"],
         "expected_status": a["expected_status"], "expected_total": a["expected_total"],
         "expected_penalties": sorted(a["expected_penalties"])}
        for a in expected["answers"]
    ])
    return df.merge(exp, on="answer_id")   # answer_id 기준으로 옆에 붙이기


# ② 정확성: 기대 점수와 얼마나 가까운가 ---------------------------
def accuracy(df):
    scored = df[df["status"] == "scored"]                       # retry 제외
    by = scored.groupby("answer_id").agg(                       # 답안별로 묶어서
        expected=("expected_total", "first"),                   #   기대 점수 (5행 모두 같음)
        ai_mean=("total", "mean"),                              #   AI 5회 평균
    )
    by["abs_err"] = (by["ai_mean"] - by["expected"]).abs()      # 오차의 절댓값
    mae = by["abs_err"].mean()                                  # MAE = 평균 절대 오차
    within10 = (by["abs_err"] <= 10).sum()                      # ±10점 이내 답안 수
    rho = spearmanr(by["expected"], by["ai_mean"]).statistic    # 순위가 얼마나 같은가
    print(f"2- 정확성  MAE {mae:.2f} | ±10 이내 {within10}/{len(by)} | 스피어만 {rho:.3f}")
    return by


# ③ 일관성: 같은 답안을 5번 채점하면 얼마나 흔들리는가 -----------
def consistency(df):
    scored = df[df["status"] == "scored"]
    by = scored.groupby("answer_id")["total"].agg(
        sd=lambda x: x.std(ddof=0),        # 표준편차 (ddof=0: 5회 전체 기준)
        spread=lambda x: x.max() - x.min() # 최대 − 최소
    )
    worst = by["sd"].idxmax()
    print(f"3- 일관성  최대 SD {by['sd'].max():.2f} ({worst}) | 최대 범위 {by['spread'].max():.0f}")
    return by


# ④ 체크포인트 일치율: 판정 하나하나가 같은가 ---------------------
def checkpoint_pairs(result, expected):
    # (답안, 체크포인트, 기대 판정, AI 판정) 쌍을 모두 만든다 → ④⑥⑦에서 같이 씀
    exp_cp = {a["answer_id"]: a["expected_checkpoints"] for a in expected["answers"]}
    pairs = []
    for r in result["records"]:
        if r["status"] != "scored":
            continue
        for cp_id, exp_label in exp_cp[r["answer_id"]].items():
            pairs.append((r["answer_id"], cp_id, exp_label, r["checkpoints"].get(cp_id)))
    return pairs


def checkpoint_agreement(pairs):
    same = sum(1 for _, _, e, a in pairs if e == a)
    print(f"4- 일치율  {same}/{len(pairs)} = {same / len(pairs) * 100:.1f}%")


# ⑤ 특수 검사: 함정 · retry · 조작 · 파싱 --------------------------
def special_checks(df):
    trap = df[df["type"] == "함정"]
    trap_ok = trap["total"].between(0, 15).sum()                       # 0~15점이면 통과
    pen_ok = sum(set(e) <= set(a) for e, a in                          # 기대 감점을 다 잡았나
                 zip(trap["expected_penalties"], trap["penalties"]))
    retry = df[df["expected_status"] == "retry"]
    retry_ok = (retry["status"] == "retry").sum()

    mean = df[df["status"] == "scored"].groupby("answer_id")["total"].mean()
    manip = {j: mean[f"{j}-07"] - mean[f"{j}-03"] for j in ["MKT", "MD", "HR"]}  # 조작 − 원본

    parse_ok = df["error"].isna().sum()
    print(f"5- 특수    함정 {trap_ok}/{len(trap)} (감점 검출 {pen_ok}/{len(trap)}) | "
          f"retry {retry_ok}/{len(retry)} | 파싱 {parse_ok}/{len(df)}")
    print("          조작 내성 (07 − 03): " + ", ".join(f"{j} {v:+.1f}" for j, v in manip.items()))


# ⑥ 불일치 분석: 어디서, 어느 방향으로 어긋났나 -------------------
def mismatch(pairs):
    wrong = [(aid, cp, e, a) for aid, cp, e, a in pairs if e != a]
    area = Counter(cp[0] for _, cp, _, _ in wrong)                     # A1 → 'A'
    direction = Counter("관대" if LEVEL[a] > LEVEL[e] else "엄격" for _, _, e, a in wrong)
    print(f"6- 불일치  {len(wrong)}건 | 영역 {dict(area.most_common())} | 방향 {dict(direction)}")
    detail = Counter(wrong)                                            # 같은 불일치가 몇 회 반복됐나
    for (aid, cp, e, a), n in sorted(detail.items()):
        print(f"          {aid} {cp}: {e} → {a} ({n}/5회)")


# ⑦ 혼동행렬 + 카파 -------------------------------------------------
def kappa_matrix(pairs):
    y_exp = [e for _, _, e, _ in pairs]
    y_ai = [a for _, _, _, a in pairs]
    cm = confusion_matrix(y_exp, y_ai, labels=LABELS)                 # 행 = 기대, 열 = AI
    k = cohen_kappa_score(y_exp, y_ai)                                 # 우연 일치를 뺀 일치도
    wk = cohen_kappa_score([LEVEL[x] for x in y_exp], [LEVEL[x] for x in y_ai],
                           weights="quadratic")                        # 두 칸 차이를 더 크게 벌점
    print(f"7- 카파    κ {k:.3f} | 가중 κ {wk:.3f}")
    print(pd.DataFrame(cm, index=[f"기대 {l}" for l in LABELS], columns=[f"AI {l}" for l in LABELS]))
    return cm


# ⑧ 그래프 3장 ------------------------------------------------------
def plot_charts(df, acc, cm):
    main, light = "#4b1f6f", "#c9c2d8"

    # 1) 산점도: 기대 점수 vs AI 평균 (대각선에 가까울수록 정확)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", linewidth=1, label="완전 일치 (y = x)")
    ax.scatter(acc["expected"], acc["ai_mean"], s=60, color=main, zorder=3, label="답안 18개")
    ax.set(xlim=(-5, 105), ylim=(-5, 105), xlabel="기대 점수", ylabel="AI 5회 평균 점수")
    ax.set_title("AI 채점 정확성: 기대 점수 vs AI 점수", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("results/1_정확성_산점도.png", dpi=200); plt.close()

    # 2) 답안별 5회 점수 분포 (점이 뭉쳐 있을수록 일관)
    scored = df[df["status"] == "scored"]
    order = acc.sort_values("expected").index                          # 기대 점수 순으로 정렬
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, aid in enumerate(order):
        runs = scored.loc[scored["answer_id"] == aid, "total"]
        ax.scatter(runs, [i] * len(runs), color=main, alpha=0.5, s=40)  # AI 5회
        ax.scatter(acc.loc[aid, "expected"], i, marker="|", s=300, color="black")  # 기대 점수
    ax.set_yticks(range(len(order)), order)
    ax.set(xlim=(-5, 105), xlabel="점수 (점 = AI 5회, 세로선 = 기대 점수)")
    ax.set_title("AI 채점 일관성: 답안별 5회 점수", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("results/2_일관성_5회분포.png", dpi=200); plt.close()

    # 3) 혼동행렬 히트맵 (대각선이 진할수록 판정 일치)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Purples")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set(xlabel="AI 판정", ylabel="기대 판정")
    ax.set_title("체크포인트 판정 혼동행렬 (1,320개)", loc="left", fontweight="bold")
    plt.tight_layout(); plt.savefig("results/3_판정_혼동행렬.png", dpi=200); plt.close()
    print("8- 그래프  results/ 에 3장 저장 완료")


# 실행 --------------------------------------------------------------
result = load("results/sol_test_21x5_1121.json")   # 105 = 21답안 × 5회
expected = load("data/test_expected.json")               # 21 = 3직군 × 7유형

df = to_table(result, expected)
print(f"1- 표      {df.shape}")
acc = accuracy(df)
consistency(df)
pairs = checkpoint_pairs(result, expected)
checkpoint_agreement(pairs)
special_checks(df)
mismatch(pairs)
cm = kappa_matrix(pairs)
plot_charts(df, acc, cm)