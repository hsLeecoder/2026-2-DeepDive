#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_preprocess.py : 따릉이 '자전거 부족 발생' 예측용 데이터셋 만들기

[입력]  data/ 폴더
  - 대여소별 대여가능 수량(1시간 단위) CSV   (data_2510.csv 등) 또는 같은 내용의 zip
  - ASOS 시간 자료 CSV                        (OBS_ASOS_TIM_*.csv)
  - 대여소 정보 엑셀                          (...25.12...xlsx)

[출력]  data/processed/
  - features.parquet            : 모델링용 표 (대여소 x 시각 1행)
  - chronic_empty_stations.csv  : 상시 비어 있는 대여소 목록 (EDA / 실행 제언용)
  - preprocess_summary.txt      : 행 수, 비율 등 요약

[목표변수 정의]
  - 재고(stock)       : CSV의 '거치대수량' 컬럼. 값의 변동 양상으로 볼 때
                        '그 시각 대여 가능한 자전거 수'로 해석한다(가정).
  - 부족(short)       : 재고 <= 2
  - at_risk           : 지금(t) 재고 >= 3  (아직 부족하지 않은 상태)
  - y_h{h}            : h시간 뒤(t+h) 재고 <= 2 이면 1, 아니면 0
                        => 모델링에서는 at_risk == 1 인 행만 사용
                           ("지금은 있는데 h시간 뒤 부족해지는가?")
  - hist_rate_h{h}    : (대여소 x 시간대 x 쉬는날여부)별 과거 부족 발생률
                        학습 구간만으로 계산(누수 방지). 베이스라인이자 참고 변수.

[시간 기준 분할]
  - test  : t >= test_start (기본 2025-12-11 00시)
  - gap   : t + 3시간이 test 구간에 걸치는 행 (누수 방지용, 모델링에서 제외)
  - train : 그 외

사용법 (저장소 루트에서):
    python scripts/01_preprocess.py
    python scripts/01_preprocess.py --sample-stations 300   # 빠른 실험용
"""
import argparse
import glob
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3)          # 몇 시간 뒤를 예측할지
SHORT_THR = 2                 # 부족 기준: 재고 <= 2
LAGS = (1, 2, 3, 6, 24, 168)  # 과거 재고 (시간 전)

# 공휴일 (분석 기간 10/1~12/31). 달력으로 한 번 확인해 보세요.
HOLIDAYS = [
    "2025-10-03",                                           # 개천절
    "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",  # 추석 연휴(+대체공휴일)
    "2025-10-09",                                           # 한글날
    "2025-12-25",                                           # 성탄절
]

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 읽기
# --------------------------------------------------------------------------
def read_csv_auto(opener, **kw):
    """UTF-8 -> CP949 순서로 인코딩을 시도해서 읽는다."""
    last = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            with opener() as f:
                return pd.read_csv(f, encoding=enc, **kw)
        except UnicodeDecodeError as e:
            last = e
    raise last


def load_stock(data_dir: Path) -> pd.DataFrame:
    sources = []
    for p in sorted(data_dir.glob("data_*.csv")):
        sources.append((p.name, (lambda p=p: open(p, "rb"))))
    if not sources:  # CSV가 없으면 zip 안의 CSV를 읽는다
        for z in sorted(data_dir.glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
            for n in names:
                sources.append((n, (lambda z=z, n=n: zipfile.ZipFile(z).open(n))))
    if not sources:
        raise SystemExit(f"[오류] {data_dir} 에서 재고 CSV(data_*.csv) 또는 zip을 찾지 못했습니다.")

    need = {"일시": "date", "대여소번호": "station_id", "대여소명": "station_name",
            "시간대": "hour", "거치대수량": "stock"}
    frames = []
    for name, opener in sources:
        df = read_csv_auto(opener, dtype={"대여소번호": str})
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"[오류] {name} 에 컬럼 {missing} 이(가) 없습니다. 실제 컬럼: {list(df.columns)}")
        frames.append(df[list(need)].rename(columns=need))
        print(f"  읽음: {name}  ({len(df):,}행)")
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int), unit="h")
    dup = df.duplicated(["station_id", "ts"]).sum()
    if dup:
        print(f"  [주의] (대여소, 시각) 중복 {dup:,}행 -> 마지막 값만 사용")
        df = df.drop_duplicates(["station_id", "ts"], keep="last")
    return df


def load_weather(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*OBS_ASOS_TIM_*.csv"))
    if not files:
        raise SystemExit(f"[오류] {data_dir} 에서 ASOS 파일(OBS_ASOS_TIM_*.csv)을 찾지 못했습니다.")
    p = files[-1]
    w = read_csv_auto(lambda: open(p, "rb"))
    ren = {"일시": "ts", "기온(°C)": "temp", "강수량(mm)": "rain", "풍속(m/s)": "wind",
           "습도(%)": "humid", "적설(cm)": "snow"}
    missing = [c for c in ren if c not in w.columns]
    if missing:
        raise SystemExit(f"[오류] ASOS 파일에 컬럼 {missing} 이(가) 없습니다. 실제 컬럼: {list(w.columns)}")
    w = w[list(ren)].rename(columns=ren)
    w["ts"] = pd.to_datetime(w["ts"])
    # 강수/적설은 '없음'일 때 빈칸으로 들어오는 경우가 대부분 -> 0으로 채움
    w["rain"] = w["rain"].fillna(0)
    w["snow"] = w["snow"].fillna(0)
    print(f"  읽음: {p.name}  ({len(w):,}행)")
    return w.drop_duplicates("ts").set_index("ts").sort_index()


def load_station_info(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*25.12*.xlsx")) or sorted(data_dir.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"[오류] {data_dir} 에서 대여소 정보 엑셀(.xlsx)을 찾지 못했습니다.")
    p = files[0]
    x = pd.read_excel(p, header=None)
    if x.shape[1] < 10:
        raise SystemExit(f"[오류] 엑셀 열이 {x.shape[1]}개입니다(10개 예상). 구조가 달라졌는지 확인하세요.")
    d = x.iloc[4:, :10].copy()  # 위쪽 4줄은 쪼개진 헤더
    d.columns = ["no", "name", "gu", "addr", "lat", "lon", "install", "lcd", "qr", "op"]
    d = d[pd.to_numeric(d["no"], errors="coerce").notna()].copy()
    d["station_id"] = d["no"].astype(float).astype(int).astype(str).str.zfill(5)
    for c in ("lat", "lon", "lcd", "qr"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # 주의: 거치대수 합계는 CSV의 재고보다 작은 경우가 많아 '참고용'으로만 둔다
    d["docks_listed"] = d["lcd"].fillna(0) + d["qr"].fillna(0)
    print(f"  읽음: {p.name}  ({len(d):,}개 대여소)")
    return d.drop_duplicates("station_id").set_index("station_id")[["gu", "lat", "lon", "docks_listed", "op"]]


# --------------------------------------------------------------------------
# 배열 도우미 (행=대여소, 열=시각)
# --------------------------------------------------------------------------
def ago(a, k):
    """k시간 전 값"""
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def ahead(a, h):
    """h시간 뒤 값"""
    out = np.full_like(a, np.nan)
    out[:, :-h] = a[:, h:]
    return out


def main():
    ap = argparse.ArgumentParser(description="따릉이 부족 발생 예측용 전처리")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=None, help="기본값: <data-dir>/processed")
    ap.add_argument("--test-start", default="2025-12-11", help="이 날짜 0시부터 test")
    ap.add_argument("--drop-zero-share", type=float, default=0.9,
                    help="재고 0 비율이 이 값보다 큰 대여소는 제외 (기본 0.9)")
    ap.add_argument("--min-hours", type=int, default=168, help="관측 시간이 이보다 적은 대여소 제외")
    ap.add_argument("--sample-stations", type=int, default=0, help="N>0이면 대여소 N개만 무작위 추출")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    print("[1/6] 파일 읽기")
    stock_df = load_stock(data_dir)
    weather = load_weather(data_dir)
    info = load_station_info(data_dir)

    # ---- 재고를 (대여소 x 시각) 격자에 올린다 ---------------------------------
    print("[2/6] 격자(대여소 x 시각) 만들기")
    t0, t1 = stock_df["ts"].min(), stock_df["ts"].max()
    full = pd.date_range(t0, t1, freq="h")
    T = len(full)
    sid_cat = pd.Categorical(stock_df["station_id"])
    all_ids = np.array(sid_cat.categories)
    ti = ((stock_df["ts"] - t0) // pd.Timedelta(hours=1)).to_numpy()
    arr = np.full((len(all_ids), T), np.nan, dtype=np.float32)
    arr[sid_cat.codes, ti] = stock_df["stock"].to_numpy(np.float32)
    names = stock_df.drop_duplicates("station_id").set_index("station_id")["station_name"]
    n_missing_ts = T - stock_df["ts"].nunique()
    del stock_df

    obs = ~np.isnan(arr)
    n_obs = obs.sum(1)
    zero_share = np.where(n_obs > 0, ((arr == 0) & obs).sum(1) / np.maximum(n_obs, 1), np.nan)

    chronic = pd.DataFrame({"station_id": all_ids, "station_name": names.reindex(all_ids).to_numpy(),
                            "zero_share": zero_share.round(4), "observed_hours": n_obs})
    chronic["dropped_from_model"] = (chronic["zero_share"] > args.drop_zero_share) | (n_obs < args.min_hours)
    chronic = chronic.join(info[["gu"]], on="station_id")
    chronic[chronic["zero_share"] > 0.5].sort_values("zero_share", ascending=False) \
        .to_csv(out_dir / "chronic_empty_stations.csv", index=False, encoding="utf-8-sig")

    keep = ~chronic["dropped_from_model"].to_numpy()
    say(f"전체 대여소 {len(all_ids):,}개 중 제외 {int((~keep).sum())}개 "
        f"(재고 0 비율 > {args.drop_zero_share} 또는 관측 {args.min_hours}시간 미만)")
    arr, ids = arr[keep], all_ids[keep]
    if args.sample_stations and args.sample_stations < len(ids):
        rng = np.random.default_rng(args.seed)
        pick = np.sort(rng.choice(len(ids), args.sample_stations, replace=False))
        arr, ids = arr[pick], ids[pick]
        say(f"무작위 {len(ids)}개 대여소만 사용 (seed={args.seed})")
    S = len(ids)
    say(f"사용 대여소 {S:,}개 x {T:,}시간 (전체 관측 중 빠진 시각 {n_missing_ts}개)")

    # ---- 시간 기준 분할 -------------------------------------------------------
    print("[3/6] 시간 분할 / 달력 변수")
    test_start = pd.Timestamp(args.test_start)
    max_h = max(HORIZONS)
    is_test_t = full >= test_start
    is_train_t = full < (test_start - pd.Timedelta(hours=max_h))
    split_t = np.where(is_test_t, "test", np.where(is_train_t, "train", "gap"))

    m = ~np.isnan(arr)                       # 재고가 관측된 칸만 행으로 만든다
    si = np.broadcast_to(np.arange(S)[:, None], (S, T))[m]
    tj = np.broadcast_to(np.arange(T)[None, :], (S, T))[m]
    n_rows = int(m.sum())

    cols = {}
    cols["station_id"] = pd.Categorical.from_codes(si, categories=ids)
    cols["ts"] = full.to_numpy()[tj]
    cols["split"] = pd.Categorical(split_t[tj], categories=["train", "gap", "test"])

    # ---- 대여소 정보 ----------------------------------------------------------
    st = info.reindex(ids)
    gu_cat = pd.Categorical(st["gu"])
    cols["gu"] = pd.Categorical.from_codes(gu_cat.codes[si], categories=gu_cat.categories)
    cols["lat"] = st["lat"].to_numpy(np.float32)[si]
    cols["lon"] = st["lon"].to_numpy(np.float32)[si]
    cols["docks_listed"] = st["docks_listed"].to_numpy(np.float32)[si]
    op_cat = pd.Categorical(st["op"])
    cols["op"] = pd.Categorical.from_codes(op_cat.codes[si], categories=op_cat.categories)

    # ---- 재고와 과거 재고 -----------------------------------------------------
    cols["stock"] = arr[m]
    for k in LAGS:
        cols[f"lag{k}"] = ago(arr, k)[m]
    cols["diff1"] = (arr - ago(arr, 1))[m]
    cols["diff3"] = (arr - ago(arr, 3))[m]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stack6 = np.stack([arr] + [ago(arr, k) for k in range(1, 6)])
        cols["mean6"] = np.nanmean(stack6, axis=0)[m]
        del stack6

    # ---- 달력 ----------------------------------------------------------------
    hour_t = full.hour.to_numpy()
    dow_t = full.dayofweek.to_numpy()
    hol_t = np.asarray(full.normalize().isin(pd.to_datetime(HOLIDAYS)))
    off_t = ((dow_t >= 5) | hol_t).astype(np.int8)
    cols["hour"] = hour_t[tj].astype(np.int8)
    cols["dow"] = dow_t[tj].astype(np.int8)
    cols["is_off_day"] = off_t[tj]
    cols["month"] = full.month.to_numpy()[tj].astype(np.int8)

    # ---- 날씨 (시각 t의 관측값) -----------------------------------------------
    wx = weather.reindex(full)
    rain3 = wx["rain"].rolling(3, min_periods=1).sum()
    tdiff3 = wx["temp"] - wx["temp"].shift(3)
    for name, series in (("temp", wx["temp"]), ("rain", wx["rain"]), ("rain_3h", rain3),
                         ("humid", wx["humid"]), ("wind", wx["wind"]), ("snow", wx["snow"]),
                         ("temp_diff_3h", tdiff3)):
        cols[name] = series.to_numpy(np.float32)[tj]

    # ---- 목표변수 -------------------------------------------------------------
    cols["at_risk"] = (cols["stock"] >= SHORT_THR + 1).astype(np.int8)
    y_grids = {}
    for h in HORIZONS:
        fut = ahead(arr, h)
        cols[f"future_stock_h{h}"] = fut[m]
        y = np.where(np.isnan(fut), np.nan, (fut <= SHORT_THR).astype(np.float32)).astype(np.float32)
        y_grids[h] = y
        cols[f"y_h{h}"] = y[m]

    # ---- 학습 구간만으로 만드는 통계 (누수 방지) -------------------------------
    print("[4/6] 학습 구간 통계 / 과거 발생률")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tr = arr[:, is_train_t]
        st_mean = np.nanmean(tr, axis=1)
        st_std = np.nanstd(tr, axis=1)
        zero_tr = np.where(np.isnan(tr), np.nan, (tr == 0).astype(np.float32))
        st_zero = np.nanmean(zero_tr, axis=1)
        st_hour = np.full((S, 24), np.nan, dtype=np.float32)
        for hr in range(24):
            sel = is_train_t & (hour_t == hr)
            if sel.any():
                st_hour[:, hr] = np.nanmean(arr[:, sel], axis=1)
    cols["st_mean_stock"] = st_mean.astype(np.float32)[si]
    cols["st_std_stock"] = st_std.astype(np.float32)[si]
    cols["st_zero_share"] = st_zero.astype(np.float32)[si]
    cols["st_hour_mean"] = st_hour[si, cols["hour"]]

    is_train_row = (cols["split"].codes == 0)
    at_risk_row = cols["at_risk"] == 1
    for h in HORIZONS:
        yv = cols[f"y_h{h}"]
        sel = is_train_row & at_risk_row & ~np.isnan(yv)
        tmp = pd.DataFrame({"si": si[sel], "hour": cols["hour"][sel],
                            "off": cols["is_off_day"][sel], "y": yv[sel]})
        H = np.full((S, 24, 2), np.nan, dtype=np.float32)
        g = tmp.groupby(["si", "hour", "off"])["y"].mean()
        gi = g.index.to_frame(index=False).to_numpy()
        H[gi[:, 0], gi[:, 1], gi[:, 2]] = g.to_numpy(np.float32)
        s_mean = np.full(S, np.nan, dtype=np.float32)
        gs = tmp.groupby("si")["y"].mean()
        s_mean[gs.index.to_numpy()] = gs.to_numpy(np.float32)
        glob_mean = float(tmp["y"].mean())
        H = np.where(np.isnan(H), s_mean[:, None, None], H)
        H = np.where(np.isnan(H), np.float32(glob_mean), H)
        cols[f"hist_rate_h{h}"] = H[si, cols["hour"], cols["is_off_day"]]

    # ---- 저장 ----------------------------------------------------------------
    print("[5/6] 저장")
    out = pd.DataFrame(cols)
    out_path = out_dir / ("features.parquet" if not args.sample_stations
                          else f"features_sample{args.sample_stations}.parquet")
    out.to_parquet(out_path, index=False)

    # ---- 요약 ----------------------------------------------------------------
    print("[6/6] 요약")
    say("")
    say(f"저장: {out_path}  ({out_path.stat().st_size / 1e6:,.0f} MB, {len(out):,}행 x {out.shape[1]}열)")
    say(f"분할(행 수): " + ", ".join(f"{k}={v:,}" for k, v in out["split"].value_counts().sort_index().items()))
    say(f"at_risk(현재 재고 >= {SHORT_THR + 1}) 비율: {out['at_risk'].mean():.3f}")
    say("")
    say("[y_h: at_risk 행 중 h시간 뒤 재고 <= 2 가 되는 비율]")
    for h in HORIZONS:
        for sp in ("train", "test"):
            d = out[(out["split"] == sp) & (out["at_risk"] == 1)][f"y_h{h}"].dropna()
            say(f"  h={h} {sp:5s}: 발생률 {d.mean():.4f}  (사건 {int(d.sum()):,}건 / {len(d):,}행)")
    say("")
    say("[결측 비율 점검]")
    for c in ["gu", "lat", "docks_listed", "temp", "rain", "lag1", "lag24", "lag168", "st_mean_stock"]:
        say(f"  {c:14s}: {out[c].isna().mean():.4f}")
    (out_dir / "preprocess_summary.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"\n완료. 요약은 {out_dir / 'preprocess_summary.txt'} 에도 저장했습니다.")


if __name__ == "__main__":
    main()
