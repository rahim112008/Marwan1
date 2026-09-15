"""
🐄 Bovine SNP Platform
Pipeline complet de bioinformatique pour puces SNP bovines.

Version 3.3 — Parser PED ADAPTATIF :
  - Mode AUTO : détection auto du format
  - Mode MANUEL : séparateur + header + colonnes configurables
  - Bouton "Analyser le fichier" : aperçu diagnostic
  - Détection d'encodage (UTF-8, UTF-16, BOM, latin-1)
  - Tous les modules : PLINK/VCF · NMF · ROH · FST pairwise · Cache
"""

import gzip
import warnings
import zipfile
from collections import Counter
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🐄 Bovine SNP Platform",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_THRESHOLDS = {
    "geno": 0.05, "mind": 0.05, "maf": 0.05, "hwe": 1e-6, "het_sd": 3.0,
}

HWE_EXACT_MAX_SNP = 20_000

ARS_UCD12_LENGTHS = {
    "1": 158_534_110, "2": 136_231_102, "3": 121_005_158, "4": 120_000_166,
    "5": 120_089_699, "6": 117_806_340, "7": 110_682_743, "8": 113_384_748,
    "9": 105_708_134, "10": 103_308_737, "11": 107_310_763, "12": 91_163_125,
    "13": 84_246_514, "14": 84_648_346, "15": 85_207_080, "16": 81_726_628,
    "17": 75_176_999, "18": 66_059_976, "19": 64_089_169, "20": 72_042_983,
    "21": 71_599_096, "22": 61_416_492, "23": 52_531_573, "24": 62_384_193,
    "25": 42_959_610, "26": 51_680_158, "27": 46_772_073, "28": 46_333_854,
    "29": 51_319_414, "X": 139_009_144, "Y": 50_927_933, "MT": 16_338,
}

BOVINE_AUTOSOMES = [str(i) for i in range(1, 30)]


# ============================================================
# [MODULE 5] CACHE STREAMLIT
# ============================================================

def _hash_ndarray(x: np.ndarray):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return ("empty",)
    return (x.shape, str(x.dtype),
            float(np.nansum(x)), float(np.nansum(np.abs(x))),
            float(np.nansum(x * x)))


HASH_FUNCS = {np.ndarray: _hash_ndarray}


def cache_data(func=None, **kw):
    def _decorate(f):
        return st.cache_data(
            show_spinner=False, hash_funcs=HASH_FUNCS, **kw)(f)
    if func is None:
        return _decorate
    return _decorate(func)


# ============================================================
# UTILITAIRES
# ============================================================

def impute_mean(gt: np.ndarray) -> np.ndarray:
    gt2 = gt.astype(np.float32, copy=True)
    col_mean = np.nanmean(gt2, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(gt2)
    if not nan_mask.any():
        return gt2
    gt2[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return gt2


def _autosome_weights():
    lens = np.array([ARS_UCD12_LENGTHS[c] for c in BOVINE_AUTOSOMES],
                    dtype=float)
    return lens / lens.sum()


def _chr_sort_key(chrom):
    s = str(chrom).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit():
        return (0, int(s), "")
    special = {"X": 100, "Y": 101, "MT": 102, "M": 102, "W": 103, "Z": 104}
    if s in special:
        return (1, special[s], "")
    return (2, 0, s)


def _detect_encoding(raw: bytes) -> str:
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return "gzip"
    if len(raw) >= 2 and raw[:2] == b"\x6c\x1b":
        return "plink_binary"
    return "text"


def _decode_bytes(raw: bytes) -> tuple:
    """
    Décode les bytes en texte avec détection d'encodage.
    Retourne (texte, encodage_utilisé).
    """
    # BOM UTF-16 LE / BE
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "UTF-16"
    # BOM UTF-8
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8"), "UTF-8 (avec BOM)"
    # Try UTF-8 strict
    try:
        return raw.decode("utf-8"), "UTF-8"
    except UnicodeDecodeError:
        pass
    # Try latin-1 (jamais d'erreur)
    try:
        return raw.decode("latin-1"), "latin-1"
    except Exception:
        return raw.decode("utf-8", errors="replace"), "UTF-8 (replace)"


# ============================================================
# PARSING MAP
# ============================================================

def parse_map(map_bytes: bytes) -> tuple:
    enc = _detect_encoding(map_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(map_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .map.gz corrompu : {e}")
    elif enc == "plink_binary":
        raise ValueError(
            "❌ Le fichier .map est en binaire PLINK (magic bytes 6C 1B).")
    else:
        text, _ = _decode_bytes(map_bytes)

    rows, rejected = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            rejected += 1
            continue
        try:
            cm = float(parts[2]) if parts[2] not in ("0", ".", "NA") else 0.0
        except ValueError:
            cm = 0.0
        try:
            bp = int(float(parts[3]))
        except (ValueError, TypeError):
            rejected += 1
            continue
        rows.append({"CHR": str(parts[0]), "SNP": parts[1],
                     "CM": cm, "BP": bp})

    if not rows:
        raise ValueError("Fichier .map vide ou invalide.")
    return pd.DataFrame(rows), rejected


# ============================================================
# PARSING PED — VERSION ADAPTATIVE v3.3
# ============================================================

SEP_OPTIONS = {
    "Auto": "auto",
    "Tabulation (\\t)": "\t",
    "Espace ( )": " ",
    "Virgule (,)": ",",
    "Point-virgule (;)": ";",
}


def _detect_separator(line: str) -> str:
    """Détecte le séparateur dominant dans une ligne."""
    counts = {
        "\t": line.count("\t"),
        " ": line.count(" "),
        ",": line.count(","),
        ";": line.count(";"),
    }
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        return " "
    return sep


def _split_line(line: str, sep: str) -> list:
    if sep == " ":
        return line.split()
    return [x.strip() for x in line.split(sep) if x.strip() != ""]


def analyze_ped_file(ped_bytes: bytes) -> dict:
    """
    Analyse préliminaire d'un fichier PED sans le parser entièrement.
    Retourne un dict avec les infos de diagnostic.
    """
    enc = _detect_encoding(ped_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(ped_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Gzip corrompu : {e}"}
        encoding_used = "GZIP → UTF-8"
    elif enc == "plink_binary":
        return {"error": "Fichier binaire PLINK (.bed renommé)"}
    else:
        text, encoding_used = _decode_bytes(ped_bytes)

    all_lines = [l.rstrip("\r\n") for l in text.splitlines()]
    non_empty = [l for l in all_lines if l.strip()]

    if not non_empty:
        return {"error": "Fichier vide"}

    # Détecter le séparateur sur les 20 premières lignes
    sep = " "
    for line in non_empty[:20]:
        s = _detect_separator(line)
        if s != " " or line.count(" ") > 3:
            sep = s
            break

    sep_name = {"\t": "tabulation", " ": "espace",
                ",": "virgule", ";": "point-virgule"}.get(sep, repr(sep))

    # Analyser les 10 premières lignes
    line_info = []
    for i, line in enumerate(non_empty[:10]):
        cols = _split_line(line, sep)
        line_info.append({
            "n": i + 1,
            "n_cols": len(cols),
            "preview": " ".join(cols[:8])[:120],
        })

    # Distribution des longueurs sur 100 premières lignes
    lengths = Counter()
    for line in non_empty[:100]:
        lengths[len(_split_line(line, sep))] += 1

    return {
        "ok": True,
        "encoding": encoding_used,
        "separator": sep,
        "separator_name": sep_name,
        "n_lines_total": len(non_empty),
        "n_lines_empty": len(all_lines) - len(non_empty),
        "line_info": line_info,
        "lengths_distribution": lengths.most_common(8),
    }


def parse_ped(ped_bytes: bytes, n_snp_map: int,
              manual_params: dict = None) -> tuple:
    """
    Parser PED ADAPTATIF v3.3.

    manual_params (optionnel) :
        {
            "separator": "auto" | "\t" | " " | "," | ";",
            "header_lines": int | "auto",   # lignes à skipper
            "metadata_cols": int,            # colonnes avant génotypes
            "expected_cols": int | "auto",   # total colonnes attendues
        }
    """
    manual_params = manual_params or {}

    # --- 1. Encodage ---
    enc = _detect_encoding(ped_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(ped_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .ped.gz corrompu : {e}")
    elif enc == "plink_binary":
        raise ValueError(
            "❌ Fichier PLINK binaire (.bed) renommé en .ped.\n"
            "👉 Convertissez : plink --bfile PREFIX --recode --out PREFIX")
    else:
        text, _ = _decode_bytes(ped_bytes)

    # --- 2. Lecture ---
    non_empty = [l.rstrip("\r\n") for l in text.splitlines() if l.strip()]
    if not non_empty:
        raise ValueError("Fichier .ped vide.")

    # --- 3. Séparateur ---
    sep_choice = manual_params.get("separator", "auto")
    if sep_choice == "auto":
        # Détection auto
        sep = " "
        for line in non_empty[:20]:
            s = _detect_separator(line)
            if s != " ":
                sep = s
                break
    else:
        sep = sep_choice

    sep_name = {"\t": "tabulation", " ": "espace",
                ",": "virgule", ";": "point-virgule"}.get(sep, repr(sep))

    # --- 4. Header ---
    header_choice = manual_params.get("header_lines", "auto")
    if header_choice == "auto":
        header_skip = 0
        for i, line in enumerate(non_empty[:50]):
            cols = _split_line(line, sep)
            if len(cols) >= 7:
                header_skip = i
                break
    else:
        header_skip = int(header_choice)

    if header_skip > 0:
        st.info(f"ℹ️ {header_skip} ligne(s) d'en-tête ignorée(s)")

    working_lines = non_empty[header_skip:]
    if not working_lines:
        raise ValueError("Toutes les lignes ont été ignorées comme header.")

    # --- 5. Colonnes ---
    meta_cols = int(manual_params.get("metadata_cols", 6))
    expected_choice = manual_params.get("expected_cols", "auto")

    # Analyser la 1ère ligne réellement utilisable
    first_cols = _split_line(working_lines[0], sep)
    n_cols_first = len(first_cols)

    if expected_choice == "auto":
        # Détecter le nombre de colonnes le plus fréquent
        lengths = Counter()
        for line in working_lines[:200]:
            lengths[len(_split_line(line, sep))] += 1
        if lengths:
            n_cols_observed = lengths.most_common(1)[0][0]
        else:
            n_cols_observed = n_cols_first

        # Le PED a meta_cols + 2*n_snp colonnes
        n_snp_ped = max(0, (n_cols_observed - meta_cols) // 2)
        expected_cols_eff = meta_cols + 2 * n_snp_ped
    else:
        expected_cols_eff = int(expected_choice)
        n_snp_ped = max(0, (expected_cols_eff - meta_cols) // 2)

    # --- 6. Ajustement ---
    if n_snp_ped != n_snp_map and n_snp_ped > 0:
        n_snp_eff = min(n_snp_map, n_snp_ped)
        st.warning(
            f"⚠️ Désalignement PED / MAP\n"
            f"   • MAP : **{n_snp_map}** SNPs\n"
            f"   • PED : **{n_snp_ped}** SNPs (colonnes : {expected_cols_eff})\n"
            f"   → Utilisation de **{n_snp_eff}** SNPs communs.")
    else:
        n_snp_eff = n_snp_map

    expected_cols_final = meta_cols + 2 * n_snp_eff
    geno_start = meta_cols  # index de début des génotypes

    # --- 7. Parse ---
    fids, iids, geno_rows = [], [], []
    rejected_short = 0
    rejected_empty = 0
    lengths_seen = Counter()

    for line in working_lines:
        cols = _split_line(line, sep)
        lengths_seen[len(cols)] += 1
        if len(cols) < meta_cols + 2:
            rejected_empty += 1
            continue
        if len(cols) < expected_cols_final:
            rejected_short += 1
            continue
        # FID et IID = 2 premières colonnes (ou fallback si meta < 2)
        if meta_cols >= 2:
            fids.append(cols[0])
            iids.append(cols[1])
        else:
            fids.append(f"IND_{len(fids)}")
            iids.append(f"IND_{len(fids)}")
        geno_rows.append(cols[geno_start:geno_start + 2 * n_snp_eff])

    n_ind = len(iids)

    # --- 8. Diagnostic si échec ---
    if n_ind == 0:
        top_lengths = lengths_seen.most_common(5)
        top_str = "\n".join(
            [f"  • {length} cols : {count} lignes"
             for length, count in top_lengths])
        sample_show = "\n".join(
            [f"  L{l+1} : {':'.join(l.split()[:8])[:100]}"
             for l in range(min(3, len(working_lines)))])
        raise ValueError(
            f"❌ Aucun individu chargé.\n\n"
            f"**Diagnostic :**\n"
            f"• Séparateur : **{sep_name}**\n"
            f"• Header skip : **{header_skip}**\n"
            f"• Méta-colonnes : **{meta_cols}**\n"
            f"• Colonnes attendues : **{expected_cols_final}**\n"
            f"• 1ère ligne : **{n_cols_first}** colonnes\n"
            f"• Lignes trop courtes : **{rejected_short}**\n"
            f"• Lignes quasi-vides : **{rejected_empty}**\n\n"
            f"**Distribution des longueurs :**\n{top_str}\n\n"
            f"**Aperçu :**\n{sample_show}\n\n"
            f"**💡 Solutions :**\n"
            f"1. Utilisez le **Mode Manuel** dans la sidebar\n"
            f"2. Changez le séparateur (Tabulation / Espace / Virgule)\n"
            f"3. Ajustez le nombre de lignes d'en-tête\n"
            f"4. Vérifiez que le .ped correspond bien au .map")

    if rejected_short > 0:
        st.warning(
            f"⚠️ {rejected_short} ligne(s) ignorée(s) — moins de "
            f"{expected_cols_final} colonnes.")

    # --- 9. Passe 1 : comptage allèles ---
    allele_counts = [Counter() for _ in range(n_snp_eff)]
    for row in geno_rows:
        for j in range(n_snp_eff):
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 in ("0", "N", ".", "") or a2 in ("0", "N", ".", ""):
                continue
            c = allele_counts[j]
            c[a1] += 1
            c[a2] += 1

    minor = [None] * n_snp_eff
    for j, c in enumerate(allele_counts):
        if len(c) < 2:
            continue
        minor[j] = min(c, key=c.get)

    # --- 10. Passe 2 : dosage 0/1/2 ---
    gt = np.full((n_ind, n_snp_eff), np.nan, dtype=np.float32)
    for i, row in enumerate(geno_rows):
        for j in range(n_snp_eff):
            m = minor[j]
            if m is None:
                continue
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 in ("0", "N", ".", "") or a2 in ("0", "N", ".", ""):
                continue
            gt[i, j] = (1 if a1 == m else 0) + (1 if a2 == m else 0)

    ind_df = pd.DataFrame({"FID": fids, "IID": iids})

    if n_snp_eff < n_snp_map:
        st.info(f"ℹ️ MAP tronqué à {n_snp_eff} SNPs (au lieu de {n_snp_map})")

    return gt, ind_df, rejected_short + rejected_empty


# ============================================================
# [MODULE 6] PARSING VCF
# ============================================================

def parse_vcf(vcf_bytes: bytes) -> tuple:
    is_gz = len(vcf_bytes) >= 2 and vcf_bytes[:2] == b"\x1f\x8b"
    if is_gz:
        try:
            text = gzip.decompress(vcf_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .vcf.gz corrompu : {e}")
    else:
        text = vcf_bytes.decode("utf-8", errors="replace")

    samples, rows = [], []
    n_skipped, n_multi = 0, 0

    for line in text.splitlines():
        line = line.rstrip("\n\r")
        if not line:
            continue
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
            continue
        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 10:
            n_skipped += 1
            continue

        chrom, pos_s, vid, ref, alt = (parts[0], parts[1], parts[2],
                                       parts[3], parts[4])
        if "," in alt:
            n_multi += 1
            continue
        try:
            pos = int(pos_s)
        except ValueError:
            n_skipped += 1
            continue

        fmt = parts[8].split(":")
        if "GT" not in fmt:
            n_skipped += 1
            continue
        gt_idx = fmt.index("GT")

        gts = np.full(len(samples), np.nan, dtype=np.float32)
        for k, s in enumerate(parts[9:]):
            fields = s.split(":")
            if gt_idx >= len(fields):
                continue
            gt_str = fields[gt_idx]
            sep = "|" if "|" in gt_str else "/"
            alleles = gt_str.split(sep)
            if any(a == "." or a == "" for a in alleles):
                continue
            try:
                gts[k] = sum(int(a) for a in alleles)
            except ValueError:
                continue

        rows.append({"CHR": str(chrom), "SNP": vid or f"{chrom}:{pos}",
                     "CM": 0.0, "BP": pos, "A1": ref, "A2": alt, "GT": gts})

    if not rows:
        raise ValueError("Aucun variant biallélique trouvé dans le VCF.")

    n_ind, n_snp = len(samples), len(rows)
    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    for j, r in enumerate(rows):
        gt[:, j] = r["GT"]

    for j in range(n_snp):
        col = gt[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        p = valid.mean() / 2.0
        if p > 0.5:
            gt[:, j] = np.where(np.isnan(col), np.nan, 2.0 - col)

    snp_df = pd.DataFrame({
        "CHR": [r["CHR"] for r in rows],
        "SNP": [r["SNP"] for r in rows],
        "CM": 0.0,
        "BP": [r["BP"] for r in rows],
        "A1": [r["A1"] for r in rows],
        "A2": [r["A2"] for r in rows],
    })
    ind_df = pd.DataFrame({"FID": samples, "IID": samples})
    return gt, ind_df, snp_df, n_skipped, n_multi


# ============================================================
# DONNÉES DE DÉMONSTRATION
# ============================================================

def generate_demo_data(n_ind=150, n_snp=800, n_pop=4, seed=42) -> tuple:
    rng = np.random.default_rng(seed)
    pool_names = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
                  "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]
    n_pop = max(1, min(n_pop, len(pool_names)))
    pop_names = pool_names[:n_pop]

    per_pop = n_ind // n_pop
    remainder = n_ind - per_pop * n_pop
    counts = [per_pop + (1 if i < remainder else 0) for i in range(n_pop)]
    n_ind_eff = sum(counts)

    p_anc = rng.beta(0.6, 0.6, n_snp)
    drift = 0.15
    gt = np.zeros((n_ind_eff, n_snp), dtype=np.float32)
    ind_rows = []
    k = 0
    for pop, n_i in zip(pop_names, counts):
        p_k = np.clip(p_anc + rng.normal(0, drift, n_snp), 0.02, 0.98)
        for i in range(n_i):
            a1 = (rng.random(n_snp) < p_k).astype(np.float32)
            a2 = (rng.random(n_snp) < p_k).astype(np.float32)
            gt[k] = a1 + a2
            ind_rows.append({"FID": pop, "IID": f"{pop}_{i + 1:03d}"})
            k += 1

    mask = rng.random(gt.shape) < 0.02
    gt[mask] = np.nan

    ind_df = pd.DataFrame(ind_rows)
    chr_names = rng.choice(BOVINE_AUTOSOMES, n_snp, p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_UCD12_LENGTHS[c])
                    for c in chr_names], dtype=np.int64)
    snp_df = pd.DataFrame({
        "CHR": chr_names,
        "SNP": [f"rs{i:07d}" for i in range(n_snp)],
        "CM": 0.0, "BP": bps,
    })
    snp_df["_k"] = snp_df["CHR"].map(_chr_sort_key)
    snp_df = (snp_df.sort_values(["_k", "BP"])
                    .drop(columns="_k").reset_index(drop=True))
    return gt, ind_df, snp_df


# ============================================================
# QC — MÉTRIQUES
# ============================================================

def missingness_per_ind(gt): return np.isnan(gt).mean(axis=1)
def missingness_per_snp(gt): return np.isnan(gt).mean(axis=0)
def allele_freq(gt): return np.nanmean(gt, axis=0) / 2.0


def maf(gt):
    p = allele_freq(gt)
    return np.minimum(p, 1.0 - p)


def heterozygosity(gt):
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt == 1, axis=1)


def hwe_exact_p(n_het, n_hom1, n_hom2):
    n = n_het + n_hom1 + n_hom2
    if n == 0:
        return np.nan
    if n_het == 0 and (n_hom1 == 0 or n_hom2 == 0):
        return 1.0
    rare = 2 * min(n_hom1, n_hom2) + n_het
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2:
        mid += 1
    probs = np.zeros(rare + 1); probs[mid] = 1.0; mysum = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        probs[ch + 2] = probs[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        mysum += probs[ch + 2]; ch += 2; chr_ -= 1; chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        probs[ch - 2] = probs[ch] * ch * (ch - 1) / (4 * (chr_ + 1) * (chc + 1))
        mysum += probs[ch - 2]; ch -= 2; chr_ += 1; chc += 1
    p_obs = probs[n_het] if n_het < len(probs) else 0.0
    return float(min(probs[probs <= p_obs + 1e-7].sum() / mysum, 1.0))


@cache_data
def hwe_pvalues(gt: np.ndarray) -> np.ndarray:
    _, n_snp = gt.shape
    pvals = np.full(n_snp, np.nan, dtype=np.float64)
    nan_mask = np.isnan(gt)
    n0 = ((gt == 0) & ~nan_mask).sum(axis=0)
    n1 = ((gt == 1) & ~nan_mask).sum(axis=0)
    n2 = ((gt == 2) & ~nan_mask).sum(axis=0)
    n_valid = n0 + n1 + n2
    if n_snp <= HWE_EXACT_MAX_SNP:
        for j in range(n_snp):
            if n_valid[j] < 5:
                continue
            pvals[j] = hwe_exact_p(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(n_valid > 0, n_valid, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_het = 2.0 * p * (1.0 - p) * n_valid
            valid = (exp_het >= 5) & np.isfinite(exp_het)
            chi2 = (np.abs(n1 - exp_het) - 0.5) ** 2 / exp_het
            pvals[valid] = 1.0 - stats.chi2.cdf(chi2[valid], df=1)
    return pvals


# ============================================================
# QC — FILTRAGE
# ============================================================

def _align_shapes(gt, ind_df, snp_df):
    n_gt, m_gt = gt.shape
    n_ind = min(n_gt, len(ind_df))
    n_snp = min(m_gt, len(snp_df))
    if (n_gt, m_gt) != (len(ind_df), len(snp_df)):
        st.warning(
            f"⚠️ Alignement corrigé — gt={gt.shape}, "
            f"ind_df={len(ind_df)}, snp_df={len(snp_df)} → "
            f"({n_ind}, {n_snp})")
    return (gt[:n_ind, :n_snp],
            ind_df.iloc[:n_ind].reset_index(drop=True),
            snp_df.iloc[:n_snp].reset_index(drop=True))


def apply_qc_filters(gt, ind_df, snp_df, params):
    gt, ind_df, snp_df = _align_shapes(gt, ind_df, snp_df)
    n0, m0 = gt.shape

    keep_snp = missingness_per_snp(gt) <= params["geno"]
    gt = gt[:, keep_snp]; snp_df = snp_df[keep_snp].reset_index(drop=True)

    keep_ind = missingness_per_ind(gt) <= params["mind"]
    gt = gt[keep_ind]; ind_df = ind_df[keep_ind].reset_index(drop=True)

    if gt.shape[1] == 0 or gt.shape[0] == 0:
        raise ValueError("Tous les SNPs ou individus exclus (missingness).")

    m = maf(gt)
    keep_maf = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, keep_maf]; snp_df = snp_df[keep_maf].reset_index(drop=True)
    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par MAF.")

    pv = hwe_pvalues(gt)
    keep_hwe = np.isnan(pv) | (pv >= params["hwe"])
    gt = gt[:, keep_hwe]; snp_df = snp_df[keep_hwe].reset_index(drop=True)
    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par HWE.")

    het = heterozygosity(gt)
    if np.nanstd(het) > 1e-9:
        z = (het - np.nanmean(het)) / np.nanstd(het)
        keep_het = np.abs(z) <= params["het_sd"]
    else:
        keep_het = np.ones_like(het, dtype=bool)
    gt = gt[keep_het]; ind_df = ind_df[keep_het].reset_index(drop=True)
    if gt.shape[0] == 0:
        raise ValueError("Tous les individus exclus par hétérozygotie.")

    return gt, ind_df, snp_df, {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]),
        "excluded_snp": int(m0 - gt.shape[1]),
    }


# ============================================================
# POPULATION GENETICS
# ============================================================

@cache_data
def fst_per_snp(gt: np.ndarray, pop_labels: np.ndarray) -> np.ndarray:
    pops = np.unique(pop_labels)
    if len(pops) < 2:
        return np.full(gt.shape[1], np.nan)
    fst = np.full(gt.shape[1], np.nan, dtype=np.float64)
    masks = {p: (pop_labels == p) for p in pops}
    for j in range(gt.shape[1]):
        p_list, n_list = [], []
        for p in pops:
            vals = gt[masks[p], j]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 3:
                continue
            p_list.append(vals.mean() / 2.0); n_list.append(len(vals))
        if len(p_list) < 2:
            continue
        p_arr, n_arr = np.asarray(p_list), np.asarray(n_list)
        p_bar = np.average(p_arr, weights=n_arr)
        h_s = np.average(2.0 * p_arr * (1.0 - p_arr), weights=n_arr)
        h_t = 2.0 * p_bar * (1.0 - p_bar)
        if h_t > 1e-9:
            fst[j] = (h_t - h_s) / h_t
    return fst


@cache_data
def fst_pairwise(gt: np.ndarray, pop_labels: np.ndarray) -> tuple:
    pops = sorted(np.unique(pop_labels))
    K = len(pops)
    matrix = np.full((K, K), np.nan)
    np.fill_diagonal(matrix, 0.0)
    masks = {p: (pop_labels == p) for p in pops}
    for i in range(K):
        for j in range(i + 1, K):
            g1 = gt[masks[pops[i]]]; g2 = gt[masks[pops[j]]]
            p1 = np.nanmean(g1, axis=0) / 2.0
            p2 = np.nanmean(g2, axis=0) / 2.0
            n1 = (~np.isnan(g1)).sum(axis=0)
            n2 = (~np.isnan(g2)).sum(axis=0)
            denom = np.where(n1 + n2 > 0, n1 + n2, np.nan)
            p_bar = (p1 * n1 + p2 * n2) / denom
            h_s1 = 2.0 * p1 * (1.0 - p1); h_s2 = 2.0 * p2 * (1.0 - p2)
            hs = (h_s1 * n1 + h_s2 * n2) / denom
            ht = 2.0 * p_bar * (1.0 - p_bar)
            with np.errstate(divide="ignore", invalid="ignore"):
                fst_j = (ht - hs) / ht
            matrix[i, j] = matrix[j, i] = float(np.nanmean(fst_j))
    return matrix, pops


@cache_data
def pca_analysis(gt: np.ndarray, n_components: int = 10) -> tuple:
    X = impute_mean(gt); X = X - X.mean(axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)
    return scores, pca.explained_variance_ratio_ * 100.0


@cache_data
def mds_analysis(gt: np.ndarray, n_components: int = 5) -> np.ndarray:
    X = impute_mean(gt); n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(axis=1) / X.shape[1]
    return SklearnMDS(
        n_components=n_components, dissimilarity="precomputed",
        random_state=42, n_init=1, max_iter=300, normalized_stress=False,
    ).fit_transform(D)


@cache_data
def ld_decay(gt: np.ndarray, snp_bp: np.ndarray,
             max_kb: float = 1000, max_snp: int = 1500,
             seed: int = 42) -> pd.DataFrame:
    n_snp = gt.shape[1]
    if n_snp < 2:
        return pd.DataFrame(columns=["dist_kb", "r2"])
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n_snp, max_snp, replace=False))
           if n_snp > max_snp else np.arange(n_snp))
    gt_sub = impute_mean(gt[:, idx])
    bp_sub = snp_bp[idx].astype(np.float64)
    X = gt_sub - gt_sub.mean(axis=0)
    std = gt_sub.std(axis=0); std[std < 1e-8] = np.nan
    X = X / std
    C = (X.T @ X) / X.shape[0]; R2 = C ** 2
    iu, ju = np.triu_indices(len(idx), k=1)
    dist_kb = (bp_sub[ju] - bp_sub[iu]) / 1000.0
    mask = (dist_kb > 0) & (dist_kb <= max_kb)
    return pd.DataFrame({"dist_kb": dist_kb[mask],
                         "r2": R2[iu[mask], ju[mask]]})


@cache_data
def kinship_matrix(gt: np.ndarray) -> np.ndarray:
    X = impute_mean(gt)
    p = np.clip(X.mean(axis=0) / 2.0, 1e-3, 1 - 1e-3)
    Z = X - 2 * p
    denom = np.sqrt(2 * p * (1 - p))
    Zn = Z / denom
    return (Zn @ Zn.T) / X.shape[1]


@cache_data
def admixture_nmf(gt: np.ndarray, K: int = 3, seed: int = 42,
                  max_iter: int = 500) -> tuple:
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    model = NMF(n_components=K, init="nndsvda", random_state=seed,
                max_iter=max_iter)
    W = model.fit_transform(X)
    s = W.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return W / s, model.components_


@cache_data
def detect_roh(gt: np.ndarray, snp_df_json: str,
               min_snps: int = 30, min_kb: float = 500.0) -> tuple:
    snp_df = pd.read_json(snp_df_json)
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    n_ind = gt.shape[0]
    rohs = []; froh_bp = np.zeros(n_ind, dtype=np.float64)
    genome_bp_total = 0.0
    for chrom in pd.unique(chr_arr):
        mask = chr_arr == chrom
        idxs = np.where(mask)[0]
        if len(idxs) < min_snps:
            continue
        order = np.argsort(bp[idxs]); idxs = idxs[order]
        chrom_span = bp[idxs[-1]] - bp[idxs[0]]
        if chrom_span < 1:
            continue
        genome_bp_total += chrom_span
        for i in range(n_ind):
            col = gt[i, idxs]; start_k = None
            for k in range(len(idxs)):
                is_hom = (col[k] == 0) or (col[k] == 2)
                if is_hom:
                    if start_k is None:
                        start_k = k
                else:
                    if start_k is not None:
                        n_run = k - start_k
                        length_kb = (bp[idxs[k - 1]]
                                     - bp[idxs[start_k]]) / 1000.0
                        if n_run >= min_snps and length_kb >= min_kb:
                            rohs.append({
                                "IID_idx": i, "CHR": chrom,
                                "start_bp": int(bp[idxs[start_k]]),
                                "end_bp": int(bp[idxs[k - 1]]),
                                "n_snp": int(n_run),
                                "length_kb": float(length_kb)})
                            froh_bp[i] += bp[idxs[k - 1]] - bp[idxs[start_k]]
                        start_k = None
            if start_k is not None:
                n_run = len(idxs) - start_k
                length_kb = (bp[idxs[-1]] - bp[idxs[start_k]]) / 1000.0
                if n_run >= min_snps and length_kb >= min_kb:
                    rohs.append({
                        "IID_idx": i, "CHR": chrom,
                        "start_bp": int(bp[idxs[start_k]]),
                        "end_bp": int(bp[idxs[-1]]),
                        "n_snp": int(n_run),
                        "length_kb": float(length_kb)})
                    froh_bp[i] += bp[idxs[-1]] - bp[idxs[start_k]]
    froh = froh_bp / genome_bp_total if genome_bp_total > 0 else froh_bp
    roh_df = (pd.DataFrame(rohs) if rohs else pd.DataFrame(
        columns=["IID_idx", "CHR", "start_bp", "end_bp", "n_snp", "length_kb"]))
    return roh_df, froh


# ============================================================
# [MODULE 1] EXPORT PLINK / VCF
# ============================================================

def _dosage_to_plink_bits(col: np.ndarray) -> np.ndarray:
    n = len(col); bits = np.zeros(n, dtype=np.uint8)
    bits[np.isnan(col)] = 0b01
    bits[col == 0] = 0b11; bits[col == 1] = 0b10; bits[col == 2] = 0b00
    return bits


def build_plink_bed(gt: np.ndarray) -> bytes:
    n_ind, n_snp = gt.shape
    n_bytes = (n_ind + 3) // 4
    buf = bytearray([0x6C, 0x1B, 0x01])
    for j in range(n_snp):
        bits = _dosage_to_plink_bits(gt[:, j])
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for k in range(n_ind):
            packed[k // 4] |= (bits[k] & 0x03) << (2 * (k % 4))
        buf.extend(packed.tobytes())
    return bytes(buf)


def build_plink_bim(snp_df):
    lines = []
    has_a1 = "A1" in snp_df.columns
    has_a2 = "A2" in snp_df.columns
    for _, r in snp_df.iterrows():
        a1 = str(r["A1"]) if has_a1 else "A"
        a2 = str(r["A2"]) if has_a2 else "G"
        if a1 in ("nan", ""): a1 = "A"
        if a2 in ("nan", "", a1): a2 = "G" if a1 == "A" else "A"
        lines.append(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t"
                     f"{int(r['BP'])}\t{a1}\t{a2}")
    return "\n".join(lines) + "\n"


def build_plink_fam(ind_df):
    lines = []
    for _, r in ind_df.iterrows():
        lines.append(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9")
    return "\n".join(lines) + "\n"


def build_vcf_output(gt, ind_df, snp_df, project="BovineSNP"):
    n_ind, n_snp = gt.shape
    has_a1 = "A1" in snp_df.columns; has_a2 = "A2" in snp_df.columns
    header = ["##fileformat=VCFv4.2",
              f"##source=BovineSNPPlatform-{project}",
              f"##fileDate={datetime.now():%Y%m%d}",
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">']
    samples = ind_df["IID"].astype(str).tolist()
    header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                  + "\t".join(samples))
    body = []
    for j in range(n_snp):
        r = snp_df.iloc[j]
        ref = str(r["A1"]) if has_a1 else "A"
        alt = str(r["A2"]) if has_a2 else "G"
        if ref in ("nan", ""): ref = "A"
        if alt in ("nan", "", ref): alt = "G" if ref == "A" else "A"
        gts = []
        for i in range(n_ind):
            v = gt[i, j]
            if np.isnan(v): gts.append("./.")
            elif v == 0: gts.append("0/0")
            elif v == 1: gts.append("0/1")
            else: gts.append("1/1")
        body.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref}\t{alt}"
                    f"\t.\tPASS\t.\tGT\t" + "\t".join(gts))
    return "\n".join(header + body) + "\n"


def build_plink_zip(gt, ind_df, snp_df, prefix="bovine_qc"):
    bed = build_plink_bed(gt); bim = build_plink_bim(snp_df)
    fam = build_plink_fam(ind_df)
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", bed)
        zf.writestr(f"{prefix}.bim", bim)
        zf.writestr(f"{prefix}.fam", fam)
    return bio.getvalue()


# ============================================================
# VISUALISATION
# ============================================================

def plot_hist(values, title, xlabel, color="#3498db"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=80, marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Fréquence", height=380,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig


def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Missingness / individu",
                                        "Missingness / SNP"))
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60,
                               marker_color="skyblue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60,
                               marker_color="coral"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=1)
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=2)
    fig.update_yaxes(title_text="Nombre", row=1, col=1)
    return fig


def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="PC1", y="PC2", color="Population",
                     title=(f"PCA — PC1 ({var_pct[0]:.1f}%) vs "
                            f"PC2 ({var_pct[1]:.1f}%)"), height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_mds(coords, labels):
    df = pd.DataFrame({"MDS1": coords[:, 0], "MDS2": coords[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population",
                     title="MDS (IBS) — Structure des populations", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_manhattan(fst, chr_col, threshold_q=0.999):
    df = pd.DataFrame({"FST": np.asarray(fst), "CHR": np.asarray(chr_col)})
    df = df.dropna(subset=["FST"]).reset_index(drop=True)
    if df.empty: return None
    order = df["CHR"].map(_chr_sort_key)
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df["_k2"] = order.map(lambda t: t[2])
    df = df.sort_values(["_k0", "_k1", "_k2"]).reset_index(drop=True)
    cumulative, ticks, labels = 0, [], []
    cum_positions = np.zeros(len(df), dtype=int)
    for chrom in df["CHR"].unique():
        sub_idx = df.index[df["CHR"] == chrom]
        s, e = cumulative, cumulative + len(sub_idx)
        cum_positions[s:e] = np.arange(s, e)
        ticks.append((s + e) / 2); labels.append(str(chrom)); cumulative = e
    df["x"] = cum_positions
    fig = go.Figure()
    unique_chrs = list(df["CHR"].unique())
    for idx_chr, chrom in enumerate(unique_chrs):
        sub = df[df["CHR"] == chrom]
        color = "#2c3e50" if idx_chr % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["FST"], mode="markers",
                                 marker=dict(size=5, color=color),
                                 name=f"chr{chrom}", showlegend=False,
                                 hoverinfo="skip"))
    q_upper = float(np.nanquantile(fst, threshold_q))
    fig.add_hline(y=q_upper, line_dash="dash", line_color="red",
                  annotation_text=(f"Top {100 * (1 - threshold_q):.1f}% = "
                                   f"{q_upper:.4f}"),
                  annotation_position="top right")
    fig.update_layout(title="Manhattan Plot — FST par SNP",
                      xaxis_title="Chromosome", yaxis_title="FST",
                      height=500, margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_ld_decay(ld_df, bin_kb=20):
    if ld_df is None or ld_df.empty: return None
    d = ld_df.copy(); d["bin"] = (d["dist_kb"] // bin_kb) * bin_kb
    agg = d.groupby("bin")["r2"].mean().reset_index()
    fig = px.line(agg, x="bin", y="r2",
                  labels={"bin": "Distance (kb)", "r2": "r² moyen"},
                  title="Déséquilibre de liaison (LD decay)", height=450)
    fig.update_traces(line=dict(color="royalblue", width=3))
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_kinship_heatmap(G, labels):
    labels = [str(x) for x in labels]; n = len(labels)
    seen, uniq = {}, []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        uniq.append(lab if seen[lab] == 1 else f"{lab}#{seen[lab]}")
    vmin = float(np.nanpercentile(G, 1)); vmax = float(np.nanpercentile(G, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -0.3, 0.5
    custom = np.stack([np.repeat(uniq, n).reshape(n, n),
                       np.tile(uniq, n).reshape(n, n)], axis=-1)
    fig = go.Figure(data=go.Heatmap(
        z=G, colorscale="RdBu", zmid=0.0, zmin=vmin, zmax=vmax,
        colorbar=dict(title="GRM"),
        hovertemplate=("Ind %{customdata[0]} × Ind %{customdata[1]}"
                       "<br>Parenté = %{z:.3f}<extra></extra>"),
        customdata=custom))
    fig.update_layout(title="Matrice de parenté (GRM)", height=650,
                      xaxis=dict(title="Individus", showticklabels=False),
                      yaxis=dict(title="Individus", showticklabels=False,
                                 autorange="reversed"),
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_admixture(Q, labels, pop_labels, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
    df["IID"] = labels; df["Pop"] = pop_labels
    df["_dom"] = Q.argmax(axis=1)
    df["_pop_order"] = pd.Categorical(pop_labels,
                                      categories=sorted(set(pop_labels)))
    df = df.sort_values(["_pop_order", "_dom"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    palette = (px.colors.qualitative.Set2
               + px.colors.qualitative.Set3)[:K]
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(
            x=df["x"], y=df[f"K{k+1}"], name=f"Composante {k+1}",
            marker_color=palette[k % len(palette)],
            hovertemplate="Ind %{customdata}<br>K" + str(k + 1)
                          + " = %{y:.2f}<extra></extra>",
            customdata=df["IID"]))
    fig.update_layout(barmode="stack",
                      title=f"Proportions d'ancestralité (K={K})",
                      xaxis_title="Individus (triés)",
                      yaxis_title="Proportion",
                      height=500, margin=dict(l=40, r=20, t=60, b=40))
    return fig, df


def plot_fst_pairwise(matrix, pops):
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=pops, y=pops, colorscale="Viridis",
        colorbar=dict(title="FST"),
        text=np.round(matrix, 4), texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>FST = %{z:.4f}<extra></extra>"))
    fig.update_layout(title="FST pairwise entre populations", height=550,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_roh_histogram(froh, labels):
    df = pd.DataFrame({"FROH": froh, "Population": labels})
    fig = px.histogram(
        df, x="FROH", color="Population", nbins=50,
        title="Distribution de FROH (fraction du génome en ROH)", height=450)
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_roh_manhattan(roh_df, ind_labels, max_ind_display=50):
    if roh_df.empty: return None
    df = roh_df.copy()
    df["IID"] = [ind_labels[i] for i in df["IID_idx"]]
    if df["IID"].nunique() > max_ind_display:
        keep = df["IID"].value_counts().head(max_ind_display).index
        df = df[df["IID"].isin(keep)]
    order = df["CHR"].map(_chr_sort_key)
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df = df.sort_values(["_k0", "_k1", "start_bp"])
    fig = go.Figure()
    chr_offset, offset = {}, 0
    for chrom in df["CHR"].unique():
        chr_len = ARS_UCD12_LENGTHS.get(str(chrom).upper(), 100_000_000)
        chr_offset[chrom] = offset; offset += chr_len
    iid_list = sorted(df["IID"].unique())
    y_map = {iid: i for i, iid in enumerate(iid_list)}
    for _, r in df.iterrows():
        x0 = chr_offset[r["CHR"]] + r["start_bp"]
        x1 = chr_offset[r["CHR"]] + r["end_bp"]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y_map[r["IID"]]] * 2, mode="lines",
            line=dict(color="crimson", width=6),
            hovertemplate=(f"Ind: {r['IID']}<br>Chr {r['CHR']}<br>"
                           f"Taille: {r['length_kb']:.0f} kb<extra></extra>"),
            showlegend=False))
    fig.update_layout(title="Carte des ROH par individu",
                      xaxis_title="Position génomique cumulée (bp)",
                      yaxis_title="Individu",
                      height=max(400, 15 * len(iid_list)),
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ============================================================
# RAPPORT HTML
# ============================================================

def build_report_html(config, stats, figures=None):
    figures = figures or {}
    first, blocks = True, []
    for title, fig in figures.items():
        if fig is None: continue
        html = fig.to_html(full_html=False,
                           include_plotlyjs="cdn" if first else False)
        first = False
        blocks.append(f"<h2>{title}</h2>{html}")
    figs_html = "\n".join(blocks) if blocks else "<p>Aucune figure.</p>"
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Rapport Bovine SNP Platform</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #222; }}
 h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 8px; }}
 h2 {{ color: #2471a3; margin-top: 28px; }}
 .summary {{ background: #f4f6f7; padding: 16px; border-radius: 8px; }}
</style></head><body>
<h1>🐄 Rapport Bovine SNP Platform</h1>
<p><b>Projet :</b> {config.get('project_name', 'N/A')} —
   <b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="summary"><h2>Résumé exécutif</h2><ul>
 <li>Individus analysés : <b>{stats['n_ind_final']}</b> (sur {stats['n_ind_init']})</li>
 <li>SNPs retenus : <b>{stats['n_snp_final']}</b> (sur {stats['n_snp_init']})</li>
 <li>Individus exclus (QC) : <b>{stats['excluded_ind']}</b></li>
 <li>SNPs exclus (QC) : <b>{stats['excluded_snp']}</b></li>
</ul></div>
{figs_html}
<hr><p style="font-size:0.85em; color:#666">
Rapport généré automatiquement par Bovine SNP Platform v3.3.</p>
</body></html>
"""


# ============================================================
# STREAMLIT — ÉTAT
# ============================================================

_STATE_KEYS = [
    "gt", "ind_df", "snp_df",
    "gt_filt", "ind_filt", "snp_filt", "qc_stats",
    "pca_scores", "pca_var", "mds_coords", "kinship",
    "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
    "admixture_Q", "admixture_K", "roh_df", "froh",
    "run_requested", "ped_analysis", "ped_bytes_cache",
    "map_bytes_cache",
]


def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None


def invalidate_downstream():
    for k in ["pca_scores", "pca_var", "mds_coords", "kinship",
              "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
              "admixture_Q", "admixture_K", "roh_df", "froh"]:
        st.session_state[k] = None


def reset_all_derived():
    for k in (["gt_filt", "ind_filt", "snp_filt", "qc_stats"]
              + ["pca_scores", "pca_var", "mds_coords", "kinship",
                 "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
                 "admixture_Q", "admixture_K", "roh_df", "froh"]):
        st.session_state[k] = None


def has_data(): return st.session_state.gt is not None


def has_qc():
    return (st.session_state.gt_filt is not None
            and st.session_state.qc_stats is not None
            and st.session_state.ind_filt is not None
            and st.session_state.snp_filt is not None)


# ============================================================
# STREAMLIT — INTERFACE
# ============================================================

def main():
    init_state()

    st.title("🐄 Bovine SNP Platform")
    st.caption("Pipeline complet pour puces SNP bovines — "
               "QC · Structure · Admixture · ROH · FST · Export PLINK/VCF.")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :",
                        ["Demo", "Upload PED/MAP", "Upload VCF"], index=0)

        if mode == "Demo":
            c1, c2 = st.columns(2)
            n_ind = c1.number_input("Individus", 20, 1000, 150, 10)
            n_snp = c2.number_input("SNPs", 100, 10000, 800, 100)
            n_pop = st.slider("Populations", 2, 10, 4)
            if st.button("🎲 Générer le jeu de démo",
                         use_container_width=True):
                with st.spinner("Génération..."):
                    gt, ind_df, snp_df = generate_demo_data(
                        int(n_ind), int(n_snp), int(n_pop))
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind_df
                    st.session_state.snp_df = snp_df
                    reset_all_derived()
                st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")

        elif mode == "Upload PED/MAP":
            st.info("💡 Formats : .ped, .ped.gz, .map, .map.gz")
            ped_file = st.file_uploader("Fichier .ped",
                                        type=["ped", "gz", "txt"])
            map_file = st.file_uploader("Fichier .map",
                                        type=["map", "gz", "txt"])

            # ========= MODE PARSING =========
            st.divider()
            st.subheader("🔧 Mode de parsing")

            parsing_mode = st.radio(
                "Mode", ["Auto (recommandé)", "Manuel"],
                index=0, key="parsing_mode")

            manual_params = None
            if parsing_mode == "Manuel":
                st.caption("Ajustez les paramètres si l'auto-détection échoue.")

                sep_label = st.selectbox(
                    "Séparateur",
                    list(SEP_OPTIONS.keys()), index=0,
                    key="manual_sep")
                sep_value = SEP_OPTIONS[sep_label]

                header_lines = st.number_input(
                    "Lignes d'en-tête à ignorer",
                    min_value=0, max_value=100, value=0, step=1,
                    key="manual_header",
                    help="Nombre de lignes à skipper avant les données")

                metadata_cols = st.number_input(
                    "Colonnes de métadonnées",
                    min_value=0, max_value=20, value=6, step=1,
                    key="manual_meta",
                    help="FID, IID, PID, MID, SEX, PHENO = 6 (standard PED)")

                use_manual_cols = st.checkbox(
                    "Forcer le nombre total de colonnes",
                    value=False, key="manual_use_cols")
                expected_cols = "auto"
                if use_manual_cols:
                    expected_cols = st.number_input(
                        "Colonnes totales attendues",
                        min_value=1, max_value=10_000_000,
                        value=100_000, step=100,
                        key="manual_expected")

                manual_params = {
                    "separator": sep_value,
                    "header_lines": int(header_lines),
                    "metadata_cols": int(metadata_cols),
                    "expected_cols": expected_cols,
                }

            # ========= ANALYSE =========
            if ped_file is not None:
                if st.button("🔍 Analyser le fichier PED",
                             use_container_width=True):
                    try:
                        with st.spinner("Analyse en cours..."):
                            ped_bytes = ped_file.read()
                            analysis = analyze_ped_file(ped_bytes)
                            st.session_state.ped_analysis = analysis
                            st.session_state.ped_bytes_cache = ped_bytes
                    except Exception as e:
                        st.error(f"❌ {e}")

                if st.session_state.ped_analysis:
                    ana = st.session_state.ped_analysis
                    if ana.get("error"):
                        st.error(f"❌ {ana['error']}")
                    else:
                        with st.expander("📊 Diagnostic du fichier",
                                         expanded=True):
                            st.write(f"**Encodage** : {ana['encoding']}")
                            st.write(f"**Séparateur détecté** : "
                                     f"`{ana['separator_name']}`")
                            st.write(f"**Lignes totales** : "
                                     f"{ana['n_lines_total']:,}")
                            st.write("**Aperçu des 10 premières lignes :**")
                            preview_df = pd.DataFrame(ana["line_info"])
                            preview_df.columns = ["Ligne", "Colonnes",
                                                  "Aperçu (8 premiers champs)"]
                            st.dataframe(preview_df, use_container_width=True)
                            st.write("**Distribution des longueurs (100 "
                                     "premières lignes) :**")
                            dist_df = pd.DataFrame(
                                ana["lengths_distribution"],
                                columns=["Colonnes", "Nb lignes"])
                            st.dataframe(dist_df, use_container_width=True)

                            # Suggestion auto
                            most_common = ana["lengths_distribution"][0][0] \
                                if ana["lengths_distribution"] else 0
                            st.info(
                                f"💡 Le format le plus fréquent a "
                                f"**{most_common}** colonnes par ligne. "
                                f"Utilisez le **Mode Manuel** pour l'imposer.")

            # ========= CHARGER =========
            if ped_file and map_file:
                if st.button("📥 Charger PED + MAP",
                             use_container_width=True, type="primary"):
                    try:
                        with st.spinner("Parsing .map..."):
                            map_df, rej_map = parse_map(map_file.read())
                        st.info(f"📋 MAP : **{len(map_df)}** SNPs")

                        # Utiliser le cache si dispo
                        if st.session_state.ped_bytes_cache:
                            ped_bytes = st.session_state.ped_bytes_cache
                        else:
                            ped_bytes = ped_file.read()

                        with st.spinner("Parsing PED..."):
                            gt, ind_df, rej_ped = parse_ped(
                                ped_bytes, len(map_df),
                                manual_params=manual_params)

                        if len(map_df) != gt.shape[1]:
                            map_df = map_df.iloc[:gt.shape[1]].reset_index(
                                drop=True)

                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = map_df
                        reset_all_derived()
                        st.session_state.ped_analysis = None
                        st.session_state.ped_bytes_cache = None

                        st.success(
                            f"✅ **{gt.shape[0]}** ind × "
                            f"**{gt.shape[1]}** SNPs chargés")
                        if rej_map or rej_ped:
                            st.warning(f"⚠️ Rejets : {rej_map} (map), "
                                       f"{rej_ped} (ped)")
                    except Exception as e:
                        st.error(f"❌ {e}")

        else:  # VCF
            vcf_file = st.file_uploader("Fichier .vcf ou .vcf.gz",
                                        type=["vcf", "gz", "txt"])
            if vcf_file is not None:
                if st.button("📥 Charger le VCF",
                             use_container_width=True):
                    try:
                        with st.spinner("Parsing VCF..."):
                            gt, ind_df, snp_df, n_skip, n_multi = parse_vcf(
                                vcf_file.read())
                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = snp_df
                        reset_all_derived()
                        msg = (f"✅ {gt.shape[0]} ind × "
                               f"{gt.shape[1]} variants")
                        if n_skip or n_multi:
                            msg += (f" — ⚠️ ignorés : {n_skip} lignes, "
                                    f"{n_multi} multialléliques")
                        st.success(msg)
                    except Exception as e:
                        st.error(f"❌ {e}")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("Missingness SNP", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["geno"], 0.01)
        mind = st.slider("Missingness individu", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["mind"], 0.01)
        maf_thr = st.slider("MAF minimal", 0.0, 0.5,
                            DEFAULT_THRESHOLDS["maf"], 0.01)
        hwe_thr = st.number_input("HWE p-value (exclure <)",
                                  value=DEFAULT_THRESHOLDS["hwe"],
                                  format="%.0e")
        het_sd = st.slider("Écart-type hétérozygotie", 1.0, 5.0,
                           DEFAULT_THRESHOLDS["het_sd"], 0.1)

        st.divider()
        if st.button("🚀 Pipeline complet", type="primary",
                     use_container_width=True):
            if not has_data():
                st.error("Chargez des données d'abord.")
            else:
                st.session_state.run_requested = True

        st.divider()
        st.caption("v3.3 — Parser adaptatif (Auto/Manuel) · PLINK/VCF · "
                   "NMF · ROH · FST · Cache · ARS-UCD1.2")

    # ---------- MAIN ----------
    if not has_data():
        st.info("👉 Générez un jeu de démo ou importez un `.ped`+`.map` "
                "ou un `.vcf` depuis la barre latérale.")

        with st.expander("💡 Comment charger un fichier PED qui ne marche pas ?",
                         expanded=False):
            st.markdown("""
            ### Étapes de diagnostic

            1. **Cliquez sur "🔍 Analyser le fichier PED"** dans la sidebar
               → vous verrez les 10 premières lignes avec leur nombre de colonnes.

            2. **Selon le résultat :**

            | Symptôme | Solution |
            |---|---|
            | 1ère ligne a **~7 colonnes** | C'est un header → mettez **1** dans "Lignes d'en-tête" |
            | Toutes les lignes ont **peu de colonnes** | Mauvais séparateur → essayez Tabulation / Virgule |
            | 1ère ligne a **~98 000 colonnes** | ✅ Format OK → laissez en Auto |
            | Lignes ont **des longueurs très variables** | Fichier tronqué → ré-uploadez |
            | Le fichier commence par `##` | C'est un **VCF** → utilisez l'onglet VCF |

            3. **Mode Manuel** dans la sidebar permet de forcer :
               - Le **séparateur** (Tab, Espace, Virgule, `;`)
               - Le nombre de **lignes d'en-tête** (0 à 100)
               - Le nombre de **colonnes de métadonnées** (défaut 6)
               - Le nombre **total de colonnes** attendues

            ### Formats supportés
            | Format | Extension |
            |---|---|
            | PLINK text | `.ped` + `.map` |
            | PLINK gzippé | `.ped.gz` + `.map.gz` |
            | VCF 4.2 | `.vcf`, `.vcf.gz` |
            | PLINK binaire | ❌ (utiliser `plink --recode`) |
            """)
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "🧬 Structure", "🎨 Admixture",
                    "📈 Démographie", "🔍 Sélection", "📤 Export",
                    "📄 Rapport"])

    # ---------- TAB 1 ----------
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())
        st.subheader("Individus")
        st.dataframe(ind_df.head(20), use_container_width=True)
        st.subheader("SNPs")
        st.dataframe(snp_df.head(5), use_container_width=True)
        with st.expander("🔬 Compatibilité ARS-UCD1.2"):
            chroms_seen = sorted(snp_df["CHR"].astype(str).unique(),
                                 key=_chr_sort_key)
            bovines = [c for c in chroms_seen
                       if c.upper().replace("CHR", "") in ARS_UCD12_LENGTHS]
            st.write(f"Chromosomes : {len(chroms_seen)}")
            st.write(f"Compatibles ARS-UCD1.2 : "
                     f"{'✅ oui' if len(bovines) == len(chroms_seen) else '⚠️ partiel'}")
            cov = []
            for c in chroms_seen:
                key = str(c).upper().replace("CHR", "")
                if key in ARS_UCD12_LENGTHS:
                    mask = snp_df["CHR"].astype(str) == c
                    max_bp = snp_df.loc[mask, "BP"].max()
                    cov.append({"CHR": c, "max_bp": int(max_bp),
                                "ARS_len": ARS_UCD12_LENGTHS[key],
                                "ok": max_bp <= ARS_UCD12_LENGTHS[key]})
            if cov:
                st.dataframe(pd.DataFrame(cov), use_container_width=True)

    # ---------- TAB 2 : QC ----------
    with tabs[1]:
        st.subheader("Contrôle qualité")
        if st.button("▶ Lancer le QC", use_container_width=True):
            try:
                with st.spinner("Filtrage QC..."):
                    params = {"geno": geno, "mind": mind, "maf": maf_thr,
                              "hwe": hwe_thr, "het_sd": het_sd}
                    gt_f, ind_f, snp_f, qc_stats = apply_qc_filters(
                        st.session_state.gt, st.session_state.ind_df,
                        st.session_state.snp_df, params)
                    st.session_state.gt_filt = gt_f
                    st.session_state.ind_filt = ind_f
                    st.session_state.snp_filt = snp_f
                    st.session_state.qc_stats = qc_stats
                    invalidate_downstream()
                st.success("✅ QC terminé.")
            except Exception as e:
                st.error(f"❌ {e}")

        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus finaux", s["n_ind_final"],
                      delta=-s["excluded_ind"], delta_color="inverse")
            c2.metric("SNPs finaux", s["n_snp_final"],
                      delta=-s["excluded_snp"], delta_color="inverse")
            c3.metric("Individus exclus", s["excluded_ind"])
            c4.metric("SNPs exclus", s["excluded_snp"])
            gt_full = st.session_state.gt
            st.plotly_chart(plot_missingness_dashboard(
                missingness_per_ind(gt_full),
                missingness_per_snp(gt_full)),
                use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_hist(maf(gt_full), "Spectre MAF",
                                          "MAF", "#2ecc71"),
                                use_container_width=True)
            with c2:
                st.plotly_chart(plot_hist(heterozygosity(gt_full),
                                          "Hétérozygotie observée",
                                          "HET", "#9b59b6"),
                                use_container_width=True)
        else:
            st.info("Cliquez sur **Lancer le QC**.")

    # ---------- TAB 3 : Structure ----------
    with tabs[2]:
        st.subheader("Structure des populations")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if st.button("▶ PCA + MDS + GRM", use_container_width=True):
                try:
                    with st.spinner("PCA..."):
                        s_, v_ = pca_analysis(st.session_state.gt_filt, 10)
                        st.session_state.pca_scores = s_
                        st.session_state.pca_var = v_
                    with st.spinner("MDS..."):
                        st.session_state.mds_coords = mds_analysis(
                            st.session_state.gt_filt, 5)
                    with st.spinner("GRM..."):
                        st.session_state.kinship = kinship_matrix(
                            st.session_state.gt_filt)
                    st.success("✅ Terminé.")
                except Exception as e:
                    st.error(f"❌ {e}")
            labels = st.session_state.ind_filt["FID"].values
            if st.session_state.pca_scores is not None:
                st.plotly_chart(plot_pca(st.session_state.pca_scores,
                                         st.session_state.pca_var, labels),
                                use_container_width=True)
            if st.session_state.mds_coords is not None:
                st.plotly_chart(plot_mds(st.session_state.mds_coords, labels),
                                use_container_width=True)
            if st.session_state.kinship is not None:
                id_labels = (st.session_state.ind_filt["FID"].astype(str)
                             + "_"
                             + st.session_state.ind_filt["IID"].astype(str)
                             ).values
                st.plotly_chart(plot_kinship_heatmap(
                    st.session_state.kinship, id_labels),
                    use_container_width=True)

    # ---------- TAB 4 : Admixture ----------
    with tabs[3]:
        st.subheader("Proportions d'ancestralité (NMF)")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            c1, c2 = st.columns([1, 2])
            K = c1.slider("Nombre d'ancestralités (K)", 2, 10, 4)
            max_iter = c2.slider("Itérations NMF", 100, 2000, 500, 100)
            if st.button("▶ Calculer l'admixture",
                         use_container_width=True):
                try:
                    with st.spinner(f"NMF K={K}..."):
                        Q, _ = admixture_nmf(
                            st.session_state.gt_filt, K=int(K),
                            max_iter=int(max_iter))
                        st.session_state.admixture_Q = Q
                        st.session_state.admixture_K = K
                    st.success(f"✅ Q : {Q.shape}")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.admixture_Q is not None:
                Q = st.session_state.admixture_Q
                K = st.session_state.admixture_K
                pops = st.session_state.ind_filt["FID"].values
                iids = st.session_state.ind_filt["IID"].values
                fig, df_sorted = plot_admixture(Q, iids, pops, K)
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Proportions moyennes par population")
                tmp = df_sorted.groupby("Pop")[
                    [f"K{k+1}" for k in range(K)]].mean()
                st.dataframe(tmp.round(3), use_container_width=True)

    # ---------- TAB 5 : Démographie ----------
    with tabs[4]:
        st.subheader("Démographie")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            sub1, sub2 = st.tabs(["LD decay", "ROH"])
            with sub1:
                c1, c2 = st.columns(2)
                max_kb = c1.slider("Distance max (kb)", 100, 5000, 1000, 100)
                max_snp = c2.slider("SNPs max", 200, 5000, 1200, 100)
                if st.button("▶ Calculer le LD", key="btn_ld",
                             use_container_width=True):
                    try:
                        with st.spinner("LD decay..."):
                            st.session_state.ld_df = ld_decay(
                                st.session_state.gt_filt,
                                st.session_state.snp_filt["BP"].values,
                                max_kb=float(max_kb),
                                max_snp=int(max_snp))
                        st.success(
                            f"✅ {len(st.session_state.ld_df):,} paires")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if (st.session_state.ld_df is not None
                        and not st.session_state.ld_df.empty):
                    fig = plot_ld_decay(st.session_state.ld_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            with sub2:
                c1, c2 = st.columns(2)
                min_snps = c1.slider("Min SNPs par ROH", 5, 200, 30, 5)
                min_kb = c2.slider("Longueur min (kb)", 50, 5000, 500, 50)
                if st.button("▶ Détecter les ROH", key="btn_roh",
                             use_container_width=True):
                    try:
                        with st.spinner("Détection ROH..."):
                            snp_json = st.session_state.snp_filt[
                                ["CHR", "BP"]].to_json()
                            roh_df, froh = detect_roh(
                                st.session_state.gt_filt, snp_json,
                                min_snps=int(min_snps),
                                min_kb=float(min_kb))
                            st.session_state.roh_df = roh_df
                            st.session_state.froh = froh
                        st.success(f"✅ {len(roh_df)} ROH détectés")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.froh is not None:
                    froh = st.session_state.froh
                    roh_df = st.session_state.roh_df
                    c1, c2, c3 = st.columns(3)
                    c1.metric("FROH moyen", f"{np.mean(froh):.4f}")
                    c2.metric("FROH médian", f"{np.median(froh):.4f}")
                    c3.metric("Total ROH", len(roh_df))
                    labels = st.session_state.ind_filt["FID"].values
                    st.plotly_chart(plot_roh_histogram(froh, labels),
                                    use_container_width=True)
                    if len(roh_df) > 0:
                        iids = st.session_state.ind_filt["IID"].values
                        top_idx = np.argsort(-froh)[:10]
                        top_df = pd.DataFrame({
                            "IID": [iids[i] for i in top_idx],
                            "Pop": [labels[i] for i in top_idx],
                            "FROH": froh[top_idx]})
                        st.subheader("Top 10 les plus consanguins")
                        st.dataframe(top_df.round(4),
                                     use_container_width=True)
                        st.plotly_chart(plot_roh_manhattan(roh_df, iids),
                                        use_container_width=True)

    # ---------- TAB 6 : Sélection ----------
    with tabs[5]:
        st.subheader("Signatures de sélection")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            sub1, sub2 = st.tabs(["FST/SNP + Manhattan", "FST pairwise"])
            with sub1:
                threshold_q = st.slider("Quantile outliers",
                                        0.95, 0.9999, 0.999, 0.0001,
                                        format="%.4f")
                if st.button("▶ FST par SNP", key="btn_fst",
                             use_container_width=True):
                    try:
                        with st.spinner("FST..."):
                            st.session_state.fst = fst_per_snp(
                                st.session_state.gt_filt,
                                st.session_state.ind_filt["FID"].values)
                        st.success("✅ FST calculé.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.fst is not None:
                    fst_clean = st.session_state.fst[
                        np.isfinite(st.session_state.fst)]
                    if len(fst_clean) > 0:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("FST moyen", f"{fst_clean.mean():.4f}")
                        c2.metric("FST médian",
                                  f"{np.median(fst_clean):.4f}")
                        c3.metric("Top outliers",
                                  f"{np.quantile(fst_clean, threshold_q):.4f}")
                        fig = plot_manhattan(
                            st.session_state.fst,
                            st.session_state.snp_filt["CHR"].values,
                            threshold_q=threshold_q)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            with sub2:
                if st.button("▶ Calculer FST pairwise", key="btn_fstp",
                             use_container_width=True):
                    try:
                        with st.spinner("FST pairwise..."):
                            mat, pops = fst_pairwise(
                                st.session_state.gt_filt,
                                st.session_state.ind_filt["FID"].values)
                            st.session_state.fst_pairwise_matrix = mat
                            st.session_state.fst_pops = pops
                        st.success("✅ Matrice calculée.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.fst_pairwise_matrix is not None:
                    st.plotly_chart(
                        plot_fst_pairwise(
                            st.session_state.fst_pairwise_matrix,
                            st.session_state.fst_pops),
                        use_container_width=True)

    # ---------- TAB 7 : Export ----------
    with tabs[6]:
        st.subheader("Export des données post-QC")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            prefix = st.text_input("Préfixe des fichiers", "bovine_qc")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("📦 Préparer PLINK ZIP",
                             use_container_width=True):
                    try:
                        with st.spinner("Génération PLINK..."):
                            data = build_plink_zip(
                                st.session_state.gt_filt,
                                st.session_state.ind_filt,
                                st.session_state.snp_filt,
                                prefix=prefix)
                            st.session_state["_plink_zip"] = data
                        st.success(f"✅ {len(data)/1024:.1f} KB prêts.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.get("_plink_zip"):
                    st.download_button(
                        "⬇ Télécharger .zip PLINK",
                        data=st.session_state["_plink_zip"],
                        file_name=f"{prefix}_plink.zip",
                        mime="application/zip",
                        use_container_width=True)
            with c2:
                if st.button("📄 Préparer VCF",
                             use_container_width=True):
                    try:
                        with st.spinner("Génération VCF..."):
                            vcf_str = build_vcf_output(
                                st.session_state.gt_filt,
                                st.session_state.ind_filt,
                                st.session_state.snp_filt,
                                project="BovineSNP")
                            st.session_state["_vcf_str"] = vcf_str
                        st.success(f"✅ {len(vcf_str)/1024:.1f} KB prêts.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.get("_vcf_str"):
                    st.download_button(
                        "⬇ Télécharger .vcf",
                        data=st.session_state["_vcf_str"].encode("utf-8"),
                        file_name=f"{prefix}.vcf",
                        mime="text/plain",
                        use_container_width=True)
            with c3:
                st.metric("Individus exportés",
                          st.session_state.gt_filt.shape[0])
                st.metric("SNPs exportés",
                          st.session_state.gt_filt.shape[1])

    # ---------- TAB 8 : Rapport ----------
    with tabs[7]:
        st.subheader("Rapport HTML")
        if not has_qc():
            st.warning("⚠️ Lancez au moins le QC.")
        else:
            project_name = st.text_input("Nom du projet", "Cattle_Project")
            if st.button("📄 Générer le rapport",
                         use_container_width=True):
                figures = {}
                if st.session_state.pca_scores is not None:
                    figures["PCA"] = plot_pca(
                        st.session_state.pca_scores,
                        st.session_state.pca_var,
                        st.session_state.ind_filt["FID"].values)
                if st.session_state.mds_coords is not None:
                    figures["MDS (IBS)"] = plot_mds(
                        st.session_state.mds_coords,
                        st.session_state.ind_filt["FID"].values)
                if st.session_state.admixture_Q is not None:
                    fig_adm, _ = plot_admixture(
                        st.session_state.admixture_Q,
                        st.session_state.ind_filt["IID"].values,
                        st.session_state.ind_filt["FID"].values,
                        st.session_state.admixture_K)
                    figures["Admixture (NMF)"] = fig_adm
                if st.session_state.fst is not None:
                    figures["Manhattan FST"] = plot_manhattan(
                        st.session_state.fst,
                        st.session_state.snp_filt["CHR"].values)
                if st.session_state.fst_pairwise_matrix is not None:
                    figures["FST pairwise"] = plot_fst_pairwise(
                        st.session_state.fst_pairwise_matrix,
                        st.session_state.fst_pops)
                if st.session_state.froh is not None:
                    figures["FROH"] = plot_roh_histogram(
                        st.session_state.froh,
                        st.session_state.ind_filt["FID"].values)
                html = build_report_html(
                    config={"project_name": project_name},
                    stats=st.session_state.qc_stats,
                    figures=figures)
                st.download_button(
                    "⬇ Télécharger le rapport",
                    data=html.encode("utf-8"),
                    file_name=(f"rapport_bovine_"
                               f"{datetime.now():%Y%m%d_%H%M}.html"),
                    mime="text/html",
                    use_container_width=True)
                st.success("✅ Rapport prêt.")
                with st.expander("Prévisualisation"):
                    st.components.v1.html(html, height=700, scrolling=True)

    # ---------- PIPELINE COMPLET ----------
    if st.session_state.get("run_requested"):
        st.session_state.run_requested = False
        try:
            with st.spinner("Pipeline complet..."):
                params = {"geno": geno, "mind": mind, "maf": maf_thr,
                          "hwe": hwe_thr, "het_sd": het_sd}
                gt_f, ind_f, snp_f, qc_stats = apply_qc_filters(
                    st.session_state.gt, st.session_state.ind_df,
                    st.session_state.snp_df, params)
                st.session_state.gt_filt = gt_f
                st.session_state.ind_filt = ind_f
                st.session_state.snp_filt = snp_f
                st.session_state.qc_stats = qc_stats
                invalidate_downstream()
                st.session_state.pca_scores, st.session_state.pca_var = \
                    pca_analysis(gt_f, 10)
                st.session_state.mds_coords = mds_analysis(gt_f, 5)
                st.session_state.kinship = kinship_matrix(gt_f)
                st.session_state.ld_df = ld_decay(
                    gt_f, snp_f["BP"].values, max_kb=1000, max_snp=1000)
                st.session_state.fst = fst_per_snp(
                    gt_f, ind_f["FID"].values)
                mat, pops = fst_pairwise(gt_f, ind_f["FID"].values)
                st.session_state.fst_pairwise_matrix = mat
                st.session_state.fst_pops = pops
                Q, _ = admixture_nmf(gt_f, K=4)
                st.session_state.admixture_Q = Q
                st.session_state.admixture_K = 4
                snp_json = snp_f[["CHR", "BP"]].to_json()
                roh_df, froh = detect_roh(gt_f, snp_json)
                st.session_state.roh_df = roh_df
                st.session_state.froh = froh
            st.success("✅ Pipeline terminé ! Consultez les onglets.")
        except Exception as e:
            st.error(f"❌ Erreur pipeline : {e}")


if __name__ == "__main__":
    main()
