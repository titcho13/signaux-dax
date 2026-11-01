# ═══════════ DAX — Signaux 15 min → 5 min + STOP volatilité (P50) — Streamlit Cloud (mobile) ═══════════
# Déploiement : GitHub + Streamlit Cloud. Secrets (optionnel) : TELEGRAM_TOKEN, TELEGRAM_CHAT_ID.
# Hébergé sur Streamlit Cloud → consultable depuis le téléphone. Historique reconstitué à chaque démarrage.
import os, io, json, time, wave, base64, html as _html, sqlite3, contextlib, warnings
import numpy as np, pandas as pd, requests, yfinance as yf
import streamlit as st
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Signaux DAX", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")   # plus lisible sur mobile

# ───────────── PARAMÈTRES ─────────────
SYMBOLE = "^GDAXI"
FUSEAU = "Europe/Paris"
HEURE_DEBUT, HEURE_FIN = "09:00", "17:30"
DERNIERE_CLOTURE_SIGNAL = "17:15"
JOURS_HISTO = 58
REFRESH_S = 20
MODE = "strict"
QUALITE_ALERTE = {"PREMIUM", "NORMAL"}
DUREE_CHRONO, SEUIL_PREMIUM, SEUIL_NORMAL = 15, 5, 10
Z_15, Z_5 = 2.0, 1.5
VOL_SEUIL = 1.3
COUT_AR_PCT = 0.00005
ANNONCES = []
BACKFILL = True
TELEGRAM = False            # activer + renseigner les secrets Streamlit
K_STOP, R_TP = 1.5, 2.0
UT_STOP = "15m"
ATR_N, JOURS_P50 = 14, 40
REF = "fenetre"
ADAPTATIF = True
VALEUR_POINT, TICK = 25, 0.5
ALARME_PREMIUM, ALARME_REPETITIONS = True, 4
SON = True
POPUP, POPUP_ALERT, POPUP_DUREE_S = True, False, 120   # alert() JS inutile sur mobile → toast à la place

# ───────────── Stockage / secrets ─────────────
# Sur Streamlit Cloud l'espace est éphémère : le journal est reconstruit (BACKFILL)
# à chaque redémarrage à partir des données Yahoo. Pour le rendre persistant,
# montez un volume persistant et fixez la variable d'env SIGNAUX_DIR dessus.
DOSSIER = os.environ.get("SIGNAUX_DIR", os.path.expanduser("~/.signaux_dax"))
os.makedirs(DOSSIER + "/seances", exist_ok=True)
DB, ETAT = f"{DOSSIER}/journal_setups_dax.db", f"{DOSSIER}/etat_alertes.json"

def _secret(k):
    try: return st.secrets[k]
    except Exception: return os.environ.get(k)
TG_TOKEN, TG_CHAT = _secret("TELEGRAM_TOKEN"), _secret("TELEGRAM_CHAT_ID")
def envoyer_telegram(txt):
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                 data={"chat_id": TG_CHAT, "text": txt}, timeout=10)

def _min(h): t = pd.Timestamp(h); return t.hour * 60 + t.minute
DEB_MIN, FIN_MIN, DER_MIN = _min(HEURE_DEBUT), _min(HEURE_FIN), _min(DERNIERE_CLOTURE_SIGNAL)
def dans_fenetre(idx): m = idx.hour * 60 + idx.minute; return (m >= DEB_MIN) & (m < FIN_MIN)
MODES = ["strict", "souple"]
FOND  = {"PREMIUM": "#1e8449", "NORMAL": "#82e0aa", "TARDIF": "#f9e79f", "EN ATTENTE": "#f5cba7", "EXPIRÉ": "#d5d8dc"}
TEXTE = {"PREMIUM": "white", "NORMAL": "black", "TARDIF": "black", "EN ATTENTE": "black", "EXPIRÉ": "#555"}
COUL = {"EN ATTENTE": "#e67e22", "PREMIUM": "#1e8449", "NORMAL": "#2471a3", "TARDIF": "#b7950b", "EXPIRÉ": "#7f8c8d"}

# ───────────── Données ─────────────
def telecharger_5m(period):
    raw = yf.download(SYMBOLE, period=period, interval="5m", auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty: raise RuntimeError("Yahoo n'a renvoyé aucune donnée")
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    return df.tz_convert(FUSEAU)

_C = {"df": None, "t": 0.0}
def donnees_5m():
    if _C["df"] is None or time.time() - _C["t"] > 1800:
        _C["df"] = telecharger_5m(f"{JOURS_HISTO}d"); _C["t"] = time.time()
    else:
        r = telecharger_5m("2d"); _C["df"] = pd.concat([_C["df"][_C["df"].index < r.index[0]], r])
    df = _C["df"]
    return df[df.index + pd.Timedelta(minutes=5) <= pd.Timestamp.now(tz=FUSEAU)]

def agreger_15m(df5):
    a = df5.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    return a[df5["close"].resample("15min").count().reindex(a.index) == 3]

# ───────────── Indicateurs / signaux ─────────────
def indicateurs(df, z_seuil, fen):
    d = df.copy()
    gap = d.index.to_series().diff() > pd.Timedelta(hours=2)
    d["ret"] = d["close"].pct_change(); d.loc[gap, "ret"] = np.nan
    pc = d["close"].shift(1).mask(gap)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N, min_periods=ATR_N).mean()
    scale = 1.4826 * d["ret"].abs().rolling(fen, min_periods=50).median()
    d["z_robuste"] = d["ret"] / scale.replace(0, np.nan)
    d["quantile_rendement"] = d["ret"].rolling(fen * 2, min_periods=100).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    vmed = d["volume"].rolling(fen, min_periods=50).median()
    d["ratio_vol"] = d["volume"] / vmed.replace(0, np.nan)
    vol_ok = d["volume"].fillna(0).sum() > 0
    d["anomalie_vol"] = (d["ratio_vol"] >= VOL_SEUIL) | (not vol_ok)
    er, el = d["close"].ewm(span=20, adjust=False).mean(), d["close"].ewm(span=50, adjust=False).mean()
    h = d["close"].resample("1h").last().dropna()
    d["ema_1h"] = h.ewm(span=20, adjust=False).mean().shift(1).reindex(d.index, method="ffill")
    d["align_score"] = np.sign(er - el) + np.sign(d["close"] - d["ema_1h"])
    a = d["align_score"]
    d["regime"] = np.select([a == 2, a == -2, a > 0, a < 0], ["HAUSSE_FORTE", "BAISSE_FORTE", "HAUSSE", "BAISSE"], "NEUTRE")
    d["momentum"] = d["close"].pct_change(3); d["score_momentum"] = np.sign(d["momentum"])
    z = d["z_robuste"]
    lp = (z >= z_seuil) & d["anomalie_vol"] & (a >= 1) & (d["momentum"] > 0)
    sp = (z <= -z_seuil) & d["anomalie_vol"] & (a <= -1) & (d["momentum"] < 0)
    d["sens_prio"] = np.where(lp, 1, np.where(sp, -1, 0))
    d["signal_prioritaire"] = np.where(lp, "ACHAT PRIORITAIRE", np.where(sp, "VENTE PRIORITAIRE", ""))
    lt = (a == 2) & (z >= 0.5) & (d["momentum"] > 0); st_ = (a == -2) & (z <= -0.5) & (d["momentum"] < 0)
    d["sens_tend"] = np.where(lt, 1, np.where(st_, -1, 0))
    for an in ANNONCES:
        t = pd.Timestamp(an).tz_localize(FUSEAU)
        m = (d.index >= t - pd.Timedelta(minutes=15)) & (d.index <= t + pd.Timedelta(minutes=30))
        d.loc[m, "sens_prio"] = 0; d.loc[m, "sens_tend"] = 0; d.loc[m, "signal_prioritaire"] = ""
    fwd = d["close"].shift(-4) / d["close"] - 1
    ok = (d["sens_prio"] != 0) & fwd.notna()
    hit = pd.Series(np.where(ok, (np.sign(fwd) == d["sens_prio"]).astype(float), np.nan), index=d.index)
    d["probabilite_directionnelle"] = hit.expanding().mean().shift(4).where(d["sens_prio"] != 0).round(2)
    return d

# ───────────── Stop volatilité (P50 de l'ATR) ─────────────
STATS = {"v": None, "atr": None}
def calculer_stats(d_ut, now):
    hist = d_ut["atr"][dans_fenetre(d_ut.index) & (d_ut.index >= now - pd.Timedelta(days=JOURS_P50))].dropna()
    if len(hist) < 30: return None
    return dict(p50=float(hist.median()), p25=float(hist.quantile(.25)), p75=float(hist.quantile(.75)), p90=float(hist.quantile(.90)),
                creneau=hist.groupby(hist.index.strftime("%H:%M")).median().to_dict())

def calcul_stop(sens, entree, t0):
    S = STATS["v"]
    if not S: return None
    dt = 5 if UT_STOP == "5m" else 15; tb = t0 - pd.Timedelta(minutes=dt)
    atr_v = STATS["atr"].asof(tb)
    if pd.isna(atr_v): atr_v = S["p50"]
    base = S["creneau"].get(f"{tb:%H:%M}", S["p50"]) if REF == "creneau" else S["p50"]
    coef = float(np.clip(atr_v / S["p50"], 0.8, 1.5)) if ADAPTATIF else 1.0
    dist = float(np.ceil(K_STOP * base * coef / TICK) * TICK)
    return dict(dist=dist, stop=entree - sens * dist, tp=entree + sens * R_TP * dist)

# ───────────── Chrono ─────────────
def construire_setups(d15, d5, jour, mode, now):
    fin5 = d5.index + pd.Timedelta(minutes=5)
    deb = pd.Timestamp(f"{jour} {HEURE_DEBUT}", tz=FUSEAU)
    s15 = d15[(d15.index.date == jour) & (d15.index >= deb) & ((d15.index.hour * 60 + d15.index.minute + 15) <= DER_MIN) & (d15["sens_prio"] != 0)]
    cle = "sens_prio" if mode == "strict" else "sens_tend"
    out = []
    for ts, r in s15.iterrows():
        sens = int(r["sens_prio"]); t0 = ts + pd.Timedelta(minutes=15); expire = t0 + pd.Timedelta(minutes=DUREE_CHRONO)
        w = d5[(fin5 > t0) & (fin5 <= expire)]
        ok = w[cle] == sens
        t_conf = delai = qualite = None
        if ok.any():
            t_conf = w.index[ok.values][0] + pd.Timedelta(minutes=5); delai = (t_conf - t0).total_seconds() / 60
            qualite = "PREMIUM" if delai <= SEUIL_PREMIUM else "NORMAL" if delai <= SEUIL_NORMAL else "TARDIF"
        etat = qualite if qualite else ("EXPIRÉ" if now > expire and (fin5.max() >= expire or now.hour * 60 + now.minute >= FIN_MIN or now.date() > jour) else "EN ATTENTE")
        entree = float(d5.loc[t_conf - pd.Timedelta(minutes=5), "close"]) if t_conf is not None else float(r["close"])
        stp = calcul_stop(sens, entree, t0) or {}
        out.append(dict(t0=t0, sens=sens, signal=r["signal_prioritaire"], prob=r["probabilite_directionnelle"], t_conf=t_conf, delai=delai,
                        qualite=qualite, etat=etat, expire=expire, entree=entree, stop=stp.get("stop"), dist=stp.get("dist"), tp=stp.get("tp")))
    return out

def txt_stop(s, court=False):
    if s.get("stop") is None: return ""
    if court: return f" → STOP {s['stop']:.1f} ({s['dist']:.1f} pts)"
    return f" | Entrée {s['entree']:.1f} → STOP {s['stop']:.1f} ({s['dist']:.1f} pts, {s['dist'] * VALEUR_POINT:.0f} €) | Obj. {R_TP:g}R {s['tp']:.1f}"

def charger_etat():
    try: return json.load(open(ETAT, encoding="utf-8"))
    except Exception: return {}

def traiter_alertes(setups, now):
    etat = charger_etat(); deja = set(etat.get("notifies", [])); nouveaux = []
    for s in setups:
        nom = "ACHAT" if s["sens"] == 1 else "VENTE"; base = f"{s['t0']:%Y-%m-%d %H:%M}|{s['sens']}"
        evts = [(base + "|new", s["t0"], f"Signal 15 min {nom}",
                 f"{s['signal']} clôturé {s['t0']:%H:%M}. Chrono démarré : même signal en 5 min avant {s['expire']:%H:%M}.")]
        if s["t_conf"] is not None and s["qualite"] in QUALITE_ALERTE:
            evts.append((base + "|conf", s["t_conf"], f"SETUP {s['qualite']} {nom}",
                         f"Confirmé en 5 min à {s['t_conf']:%H:%M} (+{s['delai']:.0f} min après {s['t0']:%H:%M}). {s['signal']}{txt_stop(s)}"))
        elif s["etat"] == "EXPIRÉ":
            evts.append((base + "|exp", s["expire"], f"Signal {nom} expiré", f"Pas de confirmation 5 min pour le signal de {s['t0']:%H:%M}."))
        for key, t_evt, titre, msg in evts:
            if key in deja: continue
            deja.add(key)
            if now - t_evt <= pd.Timedelta(minutes=30): nouveaux.append((titre, msg))
    etat["notifies"] = sorted(deja)[-300:]
    try: json.dump(etat, open(ETAT, "w", encoding="utf-8"))
    except Exception: pass
    return nouveaux

# ───────────── Sons ─────────────
def _wav(x, sr=22050):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((x * 32767).astype(np.int16).tobytes())
    return buf.getvalue()
def _ton(f, dur, sr=22050, vol=0.5):
    t = np.linspace(0, dur, int(sr * dur), False); x = vol * np.sin(2 * np.pi * f * t)
    n = int(sr * 0.01); x[:n] *= np.linspace(0, 1, n); x[-n:] *= np.linspace(1, 0, n)
    return x
def beep_wav(): return _wav(_ton(880, 0.5))
def alarme_wav():
    sr = 22050; sil = np.zeros(int(sr * 0.05))
    return _wav(np.concatenate([np.concatenate([_ton(880, 0.25, vol=0.9), sil, _ton(1320, 0.25, vol=0.9), sil]) for _ in range(ALARME_REPETITIONS)]))
def audio_html(wav):
    return f"<audio autoplay><source src='data:audio/wav;base64,{base64.b64encode(wav).decode()}' type='audio/wav'></audio>"

# ───────────── Notifications ─────────────
LOG = []
def notifier(nouveaux, now):
    for titre, msg in nouveaux:
        LOG.append(f"{now:%H:%M:%S} 🔔 {titre} — {msg}")
        if TELEGRAM and TG_TOKEN and TG_CHAT:
            try: envoyer_telegram(f"{titre}\n\n{msg}")
            except Exception as e: LOG.append(f"Telegram : {e}")

# ───────────── Journal SQLite ─────────────
COLS_J = ["jour", "t0", "sens", "mode", "final", "signal", "prob", "t_conf", "delai_min", "qualite", "etat",
          "regime_15m", "align_score", "momentum", "z_robuste", "quantile", "ratio_vol",
          "entree_ref", "stop", "tp", "dist_stop",
          "sig_entree", "sig_pnl_15m", "sig_pnl_30m", "sig_pnl_60m", "sig_mfe_60m", "sig_mae_60m", "sig_pnl_cloture", "sig_issue", "sig_pnl_stop",
          "conf_entree", "conf_pnl_15m", "conf_pnl_30m", "conf_pnl_60m", "conf_mfe_60m", "conf_mae_60m", "conf_pnl_cloture", "conf_issue", "conf_pnl_stop"]

def _v(x):
    if x is None: return None
    if isinstance(x, pd.Timestamp): return f"{x:%Y-%m-%d %H:%M}"
    if isinstance(x, (np.bool_, bool, np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)): return None if np.isnan(x) else float(x)
    return str(x)

def simuler_stop(jr, sens, e, dist, cout):
    sl, tp = e - sens * dist, e + sens * R_TP * dist
    for hi, lo in zip(jr["high"].values, jr["low"].values):
        if (lo <= sl) if sens > 0 else (hi >= sl): return "STOP", -dist - cout
        if (hi >= tp) if sens > 0 else (lo <= tp): return "TP", R_TP * dist - cout
    return "CLÔTURE", sens * (float(jr["close"].iloc[-1]) - e) - cout

def resultats(d5, t_ent, sens, p, dist=None):
    out = {f"{p}_{k}": None for k in ("entree", "pnl_15m", "pnl_30m", "pnl_60m", "mfe_60m", "mae_60m", "pnl_cloture", "issue", "pnl_stop")}
    if t_ent is None or t_ent not in d5.index: return out
    jr = d5[(d5.index >= t_ent) & (d5.index.date == t_ent.date()) & dans_fenetre(d5.index)]
    if jr.empty: return out
    e = float(jr["open"].iloc[0]); cout = COUT_AR_PCT * e; out[f"{p}_entree"] = e
    for h in (15, 30, 60):
        w = jr[jr.index < t_ent + pd.Timedelta(minutes=h)]
        if len(w): out[f"{p}_pnl_{h}m"] = sens * (float(w["close"].iloc[-1]) - e) - cout
    w = jr[jr.index < t_ent + pd.Timedelta(minutes=60)]
    hi, lo = float(w["high"].max()), float(w["low"].min())
    out[f"{p}_mfe_60m"] = (hi - e) if sens > 0 else (e - lo)
    out[f"{p}_mae_60m"] = (lo - e) if sens > 0 else (e - hi)
    out[f"{p}_pnl_cloture"] = sens * (float(jr["close"].iloc[-1]) - e) - cout
    if dist: out[f"{p}_issue"], out[f"{p}_pnl_stop"] = simuler_stop(jr, sens, e, dist, cout)
    return out

def lignes_journal(setups, d15, d5, jour, mode, final):
    rows = []
    for s in setups:
        bar = s["t0"] - pd.Timedelta(minutes=15); r = d15.loc[bar] if bar in d15.index else None
        row = dict(jour=str(jour), t0=s["t0"], sens=s["sens"], mode=mode, final=int(final), signal=s["signal"], prob=s["prob"],
                   t_conf=s["t_conf"], delai_min=s["delai"], qualite=s["qualite"], etat=s["etat"],
                   entree_ref=s["entree"], stop=s["stop"], tp=s["tp"], dist_stop=s["dist"])
        if r is not None:
            row.update(regime_15m=r["regime"], align_score=r["align_score"], momentum=r["momentum"], z_robuste=r["z_robuste"],
                       quantile=r["quantile_rendement"], ratio_vol=r["ratio_vol"])
        if final:
            row.update(resultats(d5, s["t0"], s["sens"], "sig", s["dist"])); row.update(resultats(d5, s["t_conf"], s["sens"], "conf", s["dist"]))
        rows.append({c: _v(row.get(c)) for c in COLS_J})
    return rows

def ecrire_journal(rows, ecraser=True):
    if not rows: return
    cols, ph = ", ".join(COLS_J), ", ".join("?" * len(COLS_J))
    with contextlib.closing(sqlite3.connect(DB)) as cx:
        cx.execute(f"CREATE TABLE IF NOT EXISTS setups ({cols}, PRIMARY KEY (jour, t0, sens, mode))")
        if ecraser: sql = f"INSERT OR REPLACE INTO setups ({cols}) VALUES ({ph})"
        else:
            maj = ", ".join(f"{c}=excluded.{c}" for c in COLS_J)
            sql = f"INSERT INTO setups ({cols}) VALUES ({ph}) ON CONFLICT(jour, t0, sens, mode) DO UPDATE SET {maj} WHERE setups.final = 0"
        cx.executemany(sql, [[r[c] for c in COLS_J] for r in rows]); cx.commit()

def exporter_seance(d15, d5, jour):
    for nom, d in (("15m", d15), ("5m", d5)):
        s = d[(d.index.date == jour) & dans_fenetre(d.index)]
        s.drop(columns=["ret"], errors="ignore").to_csv(f"{DOSSIER}/seances/{nom}_{jour}.csv", sep=";", decimal=",", encoding="utf-8-sig")

_G = {"sig": None, "final": None, "backfill": False}
def journaliser_seance(d15, d5, jour, now):
    jr5 = d5[(d5.index.date == jour) & dans_fenetre(d5.index)]
    if jr5.empty: return
    last = jr5.index[-1]; nm = now.hour * 60 + now.minute
    complet = last.hour * 60 + last.minute >= FIN_MIN - 5
    final = jour < now.date() or (nm >= FIN_MIN + 5 and complet)
    if final and _G["final"] == str(jour): return
    rows = []
    for m in MODES: rows += lignes_journal(construire_setups(d15, d5, jour, m, now), d15, d5, jour, m, final)
    sig = (final, tuple((r["t0"], r["sens"], r["mode"], r["etat"]) for r in rows))
    if sig == _G["sig"]: return
    ecrire_journal(rows, True); _G["sig"] = sig
    if final: exporter_seance(d15, d5, jour); _G["final"] = str(jour)

def reconstituer(d15, d5, now):
    rows = []
    for j in sorted(set(d15.index.date) & set(d5.index.date)):
        if j >= now.date(): continue
        for m in MODES: rows += lignes_journal(construire_setups(d15, d5, j, m, now), d15, d5, j, m, True)
    ecrire_journal(rows, False); return len(rows)

# ───────────── Contexte daily ─────────────
_X = {"t": 0.0, "v": None}
def contexte_jour():
    if _X["v"] and time.time() - _X["t"] < 3600: return _X["v"]
    raw = yf.download(SYMBOLE, period="2y", interval="1d", auto_adjust=False, progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    c = raw["Close"].dropna(); e50, e200 = c.ewm(span=50, adjust=False).mean().iloc[-1], c.ewm(span=200, adjust=False).mean().iloc[-1]
    er = abs(c.iloc[-1] - c.iloc[-21]) / c.diff().abs().iloc[-20:].sum()
    reg = "HAUSSIER" if c.iloc[-1] > e50 > e200 else "BAISSIER" if c.iloc[-1] < e50 < e200 else "MIXTE"
    _X.update(t=time.time(), v=dict(reg=reg, er=er, e50=e50, e200=e200, vol=c.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))); return _X["v"]

# ───────────── Rendu HTML ─────────────
def _css(etat):
    if etat not in FOND: return ""
    return f"background-color:{FOND[etat]};color:{TEXTE[etat]};font-weight:{'bold' if etat in ('PREMIUM', 'NORMAL') else 'normal'}"

def tbl(df, row_css=None):
    th = "".join(f"<th style='padding:5px 9px;background:#2c3e50;color:#fff;text-align:right;white-space:nowrap'>{_html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = "".join("<td style='padding:4px 9px;text-align:right;white-space:nowrap'>" +
                      _html.escape("" if (not isinstance(v, str) and pd.isna(v)) else str(v)) + "</td>" for v in r.values)
        body.append(f"<tr style='{row_css(r) if row_css else ''}'>{tds}</tr>")
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:.9rem;background:#f4f6f7;color:#111'>"
            f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")

def carte(titre, valeur, sous="", coul="#2c3e50"):
    return (f"<div style='flex:1;min-width:150px;padding:10px 14px;border-radius:10px;background:{coul};color:#fff;text-align:center'>"
            f"<div style='font-size:.8rem;opacity:.85'>{titre}</div><div style='font-size:1.55rem;font-weight:bold'>{valeur}</div>"
            f"<div style='font-size:.8rem;opacity:.85'>{sous}</div></div>")

def html_bandeau(setups, now, jour):
    deb = pd.Timestamp(f"{now.date()} {HEURE_DEBUT}", tz=FUSEAU)
    if jour != now.date() and now.weekday() >= 5: txt, coul = f"💤 Marché fermé — dernière séance affichée : {jour}", "#34495e"
    elif not setups and now < deb:
        mm = int((deb - now).total_seconds() // 60); txt, coul = f"⏳ Séance à partir de {HEURE_DEBUT} (dans {mm // 60:02d}h{mm % 60:02d})", "#34495e"
    elif not setups: txt, coul = "⏳ Aucun signal PRIORITAIRE en 15 min pour l'instant — en attente", "#34495e"
    else:
        att = [s for s in setups if s["etat"] == "EN ATTENTE"]; s = (att or setups)[-1]
        nom = "ACHAT" if s["sens"] == 1 else "VENTE"; ico = "🟢" if s["sens"] == 1 else "🔴"; coul = COUL[s["etat"]]
        if s["etat"] == "EN ATTENTE":
            mm, ss = divmod(int(max((s["expire"] - now).total_seconds(), 0)), 60)
            txt = f"{ico} {nom} — {s['signal']} (15 min {s['t0']:%H:%M}) — ⏱ {mm:02d}:{ss:02d} restantes — attente du même signal en 5 min"
        elif s["qualite"]:
            etoile = "⭐ " if s["qualite"] == "PREMIUM" else ""
            txt = f"{etoile}{ico} SETUP {s['qualite']} {nom} — confirmé à {s['t_conf']:%H:%M} (+{s['delai']:.0f} min) @ {s['entree']:.1f}{txt_stop(s, True)}"
        else: txt = f"{ico} {nom} — signal 15 min {s['t0']:%H:%M} EXPIRÉ (aucune confirmation 5 min)"
    return f"<div style='padding:16px;border-radius:10px;background:{coul};color:#fff;font-size:1.3rem;font-weight:bold'>{txt}</div>"

def html_stop(setups, d_ut, last_close):
    S = STATS["v"]
    if not S: return "<div style='padding:10px;background:#f4f6f7;color:#111'>Stop : historique insuffisant pour le P50.</div>"
    atr_now = float(d_ut["atr"].dropna().iloc[-1]); ratio = atr_now / S["p50"]
    coef = float(np.clip(ratio, 0.8, 1.5)) if ADAPTATIF else 1.0
    etat_vol = "calme" if ratio < 0.85 else "élevée" if ratio > 1.15 else "normale"
    att = [s for s in setups if s["etat"] == "EN ATTENTE"]; s = (att or setups)[-1] if setups else None
    if s and s["stop"] is not None:
        nom = "ACHAT" if s["sens"] == 1 else "VENTE"
        cartes = (f"<div style='display:flex;gap:10px;flex-wrap:wrap'>"
                  + carte(f"ENTRÉE {nom}", f"{s['entree']:.1f}", f"setup {s['t0']:%H:%M} — {s['etat']}", "#2c3e50")
                  + carte("STOP", f"{s['stop']:.1f}", f"{s['dist']:.1f} pts ({s['dist'] / s['entree']:.2%})", "#c0392b")
                  + carte(f"OBJECTIF {R_TP:g}R", f"{s['tp']:.1f}", f"+{R_TP * s['dist']:.1f} pts", "#1e8449")
                  + carte("RISQUE", f"{s['dist'] * VALEUR_POINT:,.0f} €", "par contrat", "#7d3c98") + "</div>")
        ref = s["entree"]
    else:
        cartes = ""; ref = last_close
    dist_i = float(np.ceil(K_STOP * S["p50"] * coef / TICK) * TICK)
    lignes = []
    for k in (1.0, 1.5, 2.0, 2.5, 3.0):
        d = float(np.ceil(k * S["p50"] * coef / TICK) * TICK)
        lignes.append({"K": k, "Distance (pts)": round(d, 1), "Risque (€)": round(d * VALEUR_POINT),
                       "Stop ACHAT": round(ref - d, 1), "Stop VENTE": round(ref + d, 1),
                       f"Obj. ACHAT {R_TP:g}R": round(ref + R_TP * d, 1), f"Obj. VENTE {R_TP:g}R": round(ref - R_TP * d, 1)})
    info = (f"<div style='padding:8px 12px;background:#f4f6f7;color:#111;border-radius:8px;margin:8px 0'>"
            f"<b>Volatilité {etat_vol}</b> : ATR({ATR_N}) {UT_STOP} = {atr_now:.2f} pts = {ratio:.2f} × P50 ({S['p50']:.2f}) | P25 {S['p25']:.2f} / P75 {S['p75']:.2f} / P90 {S['p90']:.2f}"
            f"<br>Stop = {K_STOP:g} × P50{' × coef. ' + f'{coef:.2f}' if ADAPTATIF else ''} = <b>{dist_i:.1f} pts</b> ({dist_i * VALEUR_POINT:.0f} €/contrat)"
            f" | Au dernier cours {last_close:.1f} → stop ACHAT <b>{last_close - dist_i:.1f}</b> / stop VENTE <b>{last_close + dist_i:.1f}</b></div>")
    return cartes + info + tbl(pd.DataFrame(lignes))

def tableau(d, jour, n=12):
    s = d[(d.index.date == jour) & dans_fenetre(d.index)].tail(n).iloc[::-1]
    t = s[["close", "z_robuste", "ratio_vol", "align_score", "regime", "signal_prioritaire", "probabilite_directionnelle"]].round(2).copy()
    t.insert(0, "Heure", [f"{i:%H:%M}" for i in t.index]); return t.reset_index(drop=True)

def html_setups(setups):
    if not setups: return "<div style='padding:10px;background:#f4f6f7;color:#111'>Aucun signal PRIORITAIRE en 15 min sur la séance.</div>"
    rows = []
    for s in setups[::-1]:
        star = "⭐ " if s["etat"] == "PREMIUM" else ""
        rows.append({"Clôture 15 min": f"{s['t0']:%H:%M}", "Sens": star + ("🟢 ACHAT" if s["sens"] == 1 else "🔴 VENTE"), "Signal": s["signal"], "Prob.": s["prob"],
                     "Confirmation 5 min": f"{s['t_conf']:%H:%M}" if s["t_conf"] is not None else "—",
                     "Délai (min)": f"{s['delai']:.0f}" if s["delai"] is not None else "—", "État": s["etat"],
                     "Entrée": round(s["entree"], 1), "Stop": round(s["stop"], 1) if s["stop"] is not None else "—",
                     "Dist. (pts)": round(s["dist"], 1) if s["dist"] is not None else "—",
                     "Risque (€)": round(s["dist"] * VALEUR_POINT) if s["dist"] is not None else "—",
                     f"Obj. {R_TP:g}R": round(s["tp"], 1) if s["tp"] is not None else "—"})
    return tbl(pd.DataFrame(rows), lambda r: _css(r["État"]))

def html_15m(d15, jour, setups):
    t = tableau(d15, jour); m = {f"{s['t0'] - pd.Timedelta(minutes=15):%H:%M}": s for s in setups}
    t["Validation 5 min"] = [(m[h]["etat"] + (f" {m[h]['t_conf']:%H:%M}" if m[h]["t_conf"] is not None else "")) if h in m else "" for h in t["Heure"]]
    return tbl(t, lambda r: _css(next((k for k in FOND if r["Validation 5 min"].startswith(k)), None)))

def html_historique():
    if not os.path.exists(DB): return "<div style='padding:8px;background:#f4f6f7;color:#111'>Journal vide pour l'instant.</div>"
    with contextlib.closing(sqlite3.connect(DB)) as cx: h = pd.read_sql("SELECT * FROM setups ORDER BY t0 DESC", cx)
    if h.empty: return "<div style='padding:8px;background:#f4f6f7;color:#111'>Journal vide pour l'instant.</div>"
    f = h[h["final"] == 1]
    out = (f"<div style='color:#111;background:#f4f6f7;padding:6px 10px'>{len(h)} lignes | {h['jour'].nunique()} séances | {len(f)} avec résultats | "
           f"P&L en points nets de coûts</div>")
    if len(f):
        syn = f.groupby(["mode", "etat"]).agg(n=("t0", "size"), sig_30m=("sig_pnl_30m", "mean"), sig_60m=("sig_pnl_60m", "mean"),
              conf_30m=("conf_pnl_30m", "mean"), conf_60m=("conf_pnl_60m", "mean"), conf_avec_stop=("conf_pnl_stop", "mean"),
              conf_taux_TP=("conf_issue", lambda x: (x == "TP").mean()), conf_taux_STOP=("conf_issue", lambda x: (x == "STOP").mean())).round(2).reset_index()
        out += tbl(syn, lambda r: _css(r["etat"]))
    return out

# ───────────── Cycle (exécuté à chaque rafraîchissement automatique) ─────────────
def cycle():
    now = pd.Timestamp.now(tz=FUSEAU)
    df5 = donnees_5m()
    d5 = indicateurs(df5, Z_5, 500); d15 = indicateurs(agreger_15m(df5), Z_15, 200)
    d_ut = d15 if UT_STOP == "15m" else d5
    STATS["v"] = calculer_stats(d_ut, now); STATS["atr"] = d_ut["atr"]
    jour = d5.index[-1].date()
    setups = construire_setups(d15, d5, jour, MODE, now)
    nouveaux = traiter_alertes(setups, now); notifier(nouveaux, now)
    try:
        if BACKFILL and not _G["backfill"]: _G["backfill"] = True; LOG.append(f"{now:%H:%M:%S} 🗂️ Historique reconstitué : {reconstituer(d15, d5, now)} lignes")
        journaliser_seance(d15, d5, jour, now)
    except Exception as e: LOG.append(f"Journal : {e}")
    retard = (now - (d5.index[-1] + pd.Timedelta(minutes=5))).total_seconds() / 60
    last_close = float(d5["close"].iloc[-1])
    alerte_retard = " | ⚠ retard élevé : alertes peu fiables" if retard > 15 and dans_fenetre(pd.DatetimeIndex([now]))[0] else ""
    try:
        c = contexte_jour(); j5 = d5[(d5.index.date == jour) & dans_fenetre(d5.index)]; prec = d5[d5.index.date < jour]["close"]
        var = f"{j5['close'].iloc[-1] / prec.iloc[-1] - 1:+.2%} (ouv. {j5['open'].iloc[0]:.0f})" if len(j5) and len(prec) else "—"
        a = d15["align_score"].iloc[-1]
        ctx = (f"Daily : <b>{c['reg']}</b> (ER {c['er']:.2f}, EMA50 {c['e50']:.0f} / EMA200 {c['e200']:.0f}, vol {c['vol']:.0%}) | "
               f"Variation depuis la veille : <b>{var}</b> | Tendance 15 min : <b>{d15['regime'].iloc[-1]}</b> ({'—' if pd.isna(a) else int(a)})")
    except Exception as e: ctx = f"Contexte indisponible : {_html.escape(str(e))}"
    ann = [x for x in ANNONCES if x.startswith(str(jour))]
    ann_html = f"<div style='margin-top:6px'>📅 Annonces du jour (barres filtrées) : {' | '.join(ann)}</div>" if ann else ""
    log_html = ("<div style='padding:8px 12px;background:#f4f6f7;color:#111;border-radius:8px'>" + "<br>".join(_html.escape(x) for x in LOG[-6:][::-1]) + "</div>") if LOG else ""
    page = (html_bandeau(setups, now, jour)
            + f"<div style='margin:6px 0;font-size:.85rem;color:#111'>{now:%Y-%m-%d %H:%M:%S} | {SYMBOLE} | mode {MODE} | dernière barre 5 min {d5.index[-1]:%H:%M} (retard {retard:.0f} min) | dernière 15 min {d15.index[-1]:%H:%M}{alerte_retard}</div>"
            + f"<div style='padding:8px 12px;background:#f4f6f7;border-radius:8px;color:#111'>{ctx}{ann_html}</div>"
            + "<h3 style='margin:14px 0 6px;color:#111'>🎯 Stop (volatilité P50)</h3>" + html_stop(setups, d_ut, last_close)
            + "<h3 style='margin:14px 0 6px;color:#111'>📋 Setups du jour</h3>" + html_setups(setups)
            + ("<h3 style='margin:14px 0 6px;color:#111'>🔔 Dernières alertes</h3>" + log_html if log_html else ""))
    return page, setups, nouveaux

# ───────────── Interface Streamlit ─────────────
st.markdown("<h2 style='margin:0'>📊 Signaux DAX — 15 min → 5 min</h2>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 1])
SON_ACTIF = c2.toggle("🔊 Son", help="Activer pour jouer les alertes sonores (navigateur mobile : nécessaire)")

@st.fragment(run_every=REFRESH_S)
def dashboard():
    try:
        page, setups, nouveaux = cycle()
    except Exception as e:
        st.error(f"Erreur : {e}"); return
    st.markdown(f"<div style='font-family:sans-serif'>{page}</div>", unsafe_allow_html=True)
    # Notifications visibles même sans son
    for titre, msg in nouveaux:
        st.toast(f"{titre} — {msg}", icon="🔔")
    # Son
    if SON_ACTIF and nouveaux:
        if ALARME_PREMIUM and any(t.startswith("SETUP PREMIUM") for t, _ in nouveaux):
            st.markdown(audio_html(alarme_wav()), unsafe_allow_html=True)
        elif SON:
            st.markdown(audio_html(beep_wav()), unsafe_allow_html=True)
    with st.expander("Tableaux 15 min / 5 min"):
        st.markdown("**15 min (coloré si validé par le 5 min)**")
        try:
            df5 = donnees_5m()
            d5 = indicateurs(df5, Z_5, 500); d15 = indicateurs(agreger_15m(df5), Z_15, 200)
            jour = d5.index[-1].date()
            st.markdown(html_15m(d15, jour, setups), unsafe_allow_html=True)
            st.markdown("**5 min**")
            st.markdown(tbl(tableau(d5, jour)), unsafe_allow_html=True)
        except Exception as e: st.warning(f"Tableaux indisponibles : {e}")
    with st.expander("Historique (journal)"):
        st.markdown(html_historique(), unsafe_allow_html=True)

dashboard()