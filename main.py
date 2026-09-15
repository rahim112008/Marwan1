"""
🐄 Bovine SNP Platform
Pipeline complet de bioinformatique pour puces SNP bovines.

Version 4.0.0 — Version corrigée et vectorisée
  Corrections majeures par rapport à la 3.2.1 :
    [BUG] hash de cache sur tableaux de chaînes (FID) → plus de TypeError
    [BUG] hétérozygotie biaisée par les génotypes manquants
    [BUG] incohérence allèles/dosage à l'export (VCF flip non répercuté)
    [BUG] detect_roh via pd.read_json (déprécié + risque de désalignement)
    [BUG] heuristique "format transposé" trop agressive
    [BUG] choix de l'allèle mineur non déterministe en cas d'égalité
  Performance :
    parser PED vectorisé (~100x), ROH vectorisé (~100x),
    FST/SNP vectorisé, écriture .bed vectorisée, MDS par produits
    matriciels (BLAS), tracé ROH en une seule trace Plotly.

Convention interne : le dosage 0/1/2 compte l'allèle A1 (dosage 2 = A1/A1).
"""

import gzip
import inspect
import hashlib
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

APP_VERSION = "4.0.0"

# Compatibilité Streamlit : use_container_width est déprécié au profit de
# width="stretch" (>= 1.49). On choisit automatiquement selon la version.
try:
    _ST_VER = tuple(int(p) for p in st.__version__.split(".")[:2])
except Exception:
    _ST_VER = (1, 0)
STRETCH = ({"width": "stretch"} if _ST_VER >= (1, 49)
           else {"use_container_width": True})

DEFAULT_THRESHOLDS = {
    "geno": 0.05, "mind": 0.05, "maf": 0.05, "hwe": 1e-6, "het_sd": 3.0,
}

HWE_EXACT_MAX_SNP = 20_000
MDS_MAX_SNP = 5_000          # sous-échantillonnage pour la distance IBS
MISSING_ALLELES = {"0", ".", "N", "n", "-", "NA", "na", ""}

# Longueurs chromosomiques ARS-UCD1.2 (bp)
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
# CACHE STREAMLIT
# ============================================================

def _hash_ndarray(x):
    """Hash robuste : gère float, int, bool ET chaînes/objets.

    [CORRECTION v4] l'ancienne version appelait np.nansum() sur des
    tableaux de chaînes (ex. les FID passés à fst_per_snp) → TypeError.
    """
    if not isinstance(x, np.ndarray):
        return ("not_ndarray", repr(type(x)))
    if x.size == 0:
        return ("empty", x.shape, str(x.dtype))
    if x.dtype.kind in "fc":
        finite = np.isfinite(x)
        return (
            x.shape, str(x.dtype), int(finite.sum()),
            float(np.nansum(np.where(finite, x, 0.0))),
            float(np.nansum(np.where(finite, np.abs(x), 0.0))),
            float(np.nansum(np.where(finite, x * x, 0.0))),
        )
    if x.dtype.kind in "iub":
        return (x.shape, str(x.dtype),
                hashlib.md5(np.ascontiguousarray(x)).hexdigest())
    # chaînes, objets, dates…
    flat = np.asarray(x).ravel().astype(str)
    h = hashlib.md5("\x1f".join(flat.tolist()).encode("utf-8")).hexdigest()
    return (x.shape, str(x.dtype), h)


HASH_FUNCS = {np.ndarray: _hash_ndarray}


def cache_data(func=None, **kw):
    """Décorateur flexible : @cache_data ou @cache_data(ttl=3600)."""
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
    """Impute les NaN par la moyenne de la colonne (0 si colonne vide)."""
    gt2 = np.asarray(gt, dtype=np.float32)
    nan_mask = np.isnan(gt2)
    if not nan_mask.any():
        return gt2.copy()
    gt2 = gt2.copy()
    col_mean = np.nanmean(np.where(nan_mask, np.nan, gt2), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    gt2[nan_mask] = np.take(col_mean.astype(np.float32),
                            np.where(nan_mask)[1])
    return gt2


def as_str_array(x) -> np.ndarray:
    """Force un vrai ndarray NumPy de chaînes.

    [CORRECTION v4] avec pandas >= 2.2 / 3.0, Series.values sur une colonne
    texte renvoie un ArrowStringArray : ce n'est PAS un ndarray, donc les
    hash_funcs du cache Streamlit ne s'appliquent pas et @st.cache_data
    lève UnhashableParamError. On normalise systématiquement avant tout
    appel à une fonction cachée.
    """
    return np.asarray(np.asarray(x, dtype=object), dtype=str)


def as_int_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


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
    """Détecte si un fichier est gzip, binaire PLINK ou texte."""
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return "gzip"
    if len(raw) >= 2 and raw[:2] == b"\x6c\x1b":
        return "plink_binary"
    return "text"


def _decode(raw: bytes, label: str) -> str:
    enc = _detect_encoding(raw)
    if enc == "gzip":
        try:
            return gzip.decompress(raw).decode("utf-8", errors="replace")
        except Exception as exc:
            raise ValueError(f"Fichier {label}.gz corrompu : {exc}") from exc
    if enc == "plink_binary":
        raise ValueError(
            f"❌ Ce fichier {label} est un binaire PLINK (.bed) renommé.\n"
            "👉 Convertissez-le d'abord :\n"
            "   plink --bfile PREFIX --recode --out PREFIX")
    return raw.decode("utf-8", errors="replace")


def _ensure_alleles(snp_df: pd.DataFrame) -> pd.DataFrame:
    """Garantit des colonnes A1/A2 exploitables et distinctes."""
    out = snp_df.copy()
    if "A1" not in out.columns:
        out["A1"] = "A"
    if "A2" not in out.columns:
        out["A2"] = "G"
    a1 = out["A1"].astype(str).str.strip()
    a2 = out["A2"].astype(str).str.strip()
    bad = a1.isin(["", "nan", "None", "0", "."])
    a1 = a1.mask(bad, "A")
    bad2 = a2.isin(["", "nan", "None", "0", "."]) | (a2 == a1)
    a2 = a2.mask(bad2, np.where(a1 == "A", "G", "A"))
    out["A1"], out["A2"] = a1, a2
    return out


# ============================================================
# PARSING MAP
# ============================================================

def parse_map(map_bytes: bytes) -> tuple:
    """Parse un fichier .map (chr, snp_id, cm, bp). Gère gzip."""
    text = _decode(map_bytes, ".map")

    chrs, snps, cms, bps = [], [], [], []
    rejected = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
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
        chrs.append(str(parts[0]))
        snps.append(parts[1])
        cms.append(cm)
        bps.append(bp)

    if not chrs:
        raise ValueError("Fichier .map vide ou invalide.")
    df = pd.DataFrame({"CHR": chrs, "SNP": snps, "CM": cms, "BP": bps})
    return df, rejected


# ============================================================
# PARSING PED — VERSION VECTORISÉE
# ============================================================

def _diagnose_ped_first_line(line: str, n_snp_map: int) -> dict:
    parts = line.split()
    n_cols = len(parts)
    return {
        "n_cols": n_cols,
        "first_cols": parts[:10],
        "expected_cols": 6 + 2 * n_snp_map,
        "n_snp_inferred": max(0, (n_cols - 6) // 2),
    }


def parse_ped(ped_bytes: bytes, n_snp_map: int) -> tuple:
    """Parser PED universel, robuste et vectorisé.

    Gère : PED texte, .ped.gz, binaire PLINK renommé (erreur claire),
    désalignement PED/MAP (auto-ajustement), fichier tronqué.

    Retourne : (gt, ind_df, allele_df, n_rejected)
      gt        : dosage float32 (n_ind, n_snp), NaN = manquant,
                  2 = homozygote allèle A1 (allèle mineur observé)
      allele_df : colonnes A1 (compté) et A2
    """
    text = _decode(ped_bytes, ".ped")
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
    if not lines:
        raise ValueError("Fichier .ped vide.")

    diag = _diagnose_ped_first_line(lines[0], n_snp_map)
    n_cols_first = diag["n_cols"]
    n_snp_ped = diag["n_snp_inferred"]

    if n_cols_first < 7:
        raise ValueError(
            f"❌ Format PED invalide.\n"
            f"   La 1ère ligne contient seulement {n_cols_first} colonnes.\n"
            f"   Un PED standard a au minimum 7 colonnes "
            f"(FID, IID, PID, MID, SEX, PHENO + génotypes).\n"
            f"   → Vérifiez le séparateur (espaces/tabulations).")

    # [CORRECTION v4] heuristique "transposé" resserrée : elle ne se
    # déclenche plus sur un petit panel de marqueurs de parenté.
    if (n_snp_ped < 10 and n_snp_map > 100
            and len(lines) > max(1000, 10 * n_cols_first)):
        raise ValueError(
            f"⚠️ Fichier probablement au format TRANSPOSÉ (--tfile).\n"
            f"   {len(lines)} lignes × {n_cols_first} colonnes alors que le "
            f"MAP annonce {n_snp_map} SNPs.\n"
            f"   👉 plink --tfile PREFIX --recode --out PREFIX")

    if n_snp_ped != n_snp_map and n_snp_ped > 0:
        st.warning(
            f"⚠️ Désalignement PED / MAP détecté — MAP : **{n_snp_map}** "
            f"SNPs, PED : **{n_snp_ped}** SNPs (1ère ligne). "
            f"→ **{min(n_snp_map, n_snp_ped)}** SNPs communs utilisés.")
        n_snp_eff = min(n_snp_map, n_snp_ped)
    else:
        n_snp_eff = n_snp_map

    if n_snp_eff < 1:
        raise ValueError("Aucun SNP exploitable (PED/MAP incompatibles).")

    n_geno_cols = 2 * n_snp_eff
    expected_cols_eff = 6 + n_geno_cols

    # ---- Passe 1 : sélection des lignes valides + inventaire allèles ----
    fids, iids, keep_idx = [], [], []
    rejected_short = rejected_empty = 0
    lengths_seen = Counter()
    allele_set = set()

    for li, line in enumerate(lines):
        parts = line.split()
        lengths_seen[len(parts)] += 1
        if len(parts) < 7:
            rejected_empty += 1
            continue
        if len(parts) < expected_cols_eff:
            rejected_short += 1
            continue
        fids.append(parts[0])
        iids.append(parts[1])
        keep_idx.append(li)
        allele_set.update(parts[6:expected_cols_eff])

    n_ind = len(keep_idx)

    if n_ind == 0:
        top_str = ", ".join(f"{length} cols × {count} lignes"
                            for length, count in lengths_seen.most_common(3))
        raise ValueError(
            "❌ Aucun individu chargé.\n\n"
            "**Diagnostic automatique :**\n"
            f"• SNPs dans le MAP : **{n_snp_map}**\n"
            f"• Colonnes attendues par ligne : **{expected_cols_eff}**\n"
            f"• 1ère ligne PED : **{n_cols_first}** colonnes\n"
            f"• Longueurs les plus fréquentes : {top_str}\n"
            f"• Lignes trop courtes : {rejected_short}\n"
            f"• Lignes quasi-vides : {rejected_empty}\n\n"
            "**Causes probables :** fichier tronqué, PED/MAP non "
            "correspondants, séparateur inattendu, ou format non PLINK.")

    if rejected_short:
        st.warning(f"⚠️ {rejected_short} ligne(s) ignorée(s) — moins de "
                   f"{expected_cols_eff} colonnes.")

    # ---- Encodage entier des allèles (vectorisé) ----
    uniq = np.array(sorted(allele_set))
    n_all = len(uniq)
    missing_codes = np.array(
        [i for i, a in enumerate(uniq) if a in MISSING_ALLELES], dtype=np.int64)

    codes = np.empty((n_ind, n_geno_cols), dtype=np.int16)
    for i, li in enumerate(keep_idx):
        parts = lines[li].split()
        codes[i] = np.searchsorted(uniq, np.asarray(parts[6:expected_cols_eff]))

    a1c = codes[:, 0::2]
    a2c = codes[:, 1::2]
    del codes

    if missing_codes.size:
        missing = (np.isin(a1c, missing_codes) | np.isin(a2c, missing_codes))
    else:
        missing = np.zeros(a1c.shape, dtype=bool)
    valid = ~missing

    # Comptage des allèles par SNP (n_alleles × n_snp)
    cnt = np.zeros((n_all, n_snp_eff), dtype=np.int64)
    missing_set = set(missing_codes.tolist())
    for c in range(n_all):
        if c in missing_set:
            continue
        cnt[c] = ((a1c == c) & valid).sum(axis=0) + \
                 ((a2c == c) & valid).sum(axis=0)

    # [CORRECTION v4] choix déterministe : plus petit effectif, puis plus
    # petit code allélique en cas d'égalité (argmin renvoie le 1er).
    cnt_min = cnt.astype(np.float64)
    cnt_min[cnt == 0] = np.inf
    minor_code = np.argmin(cnt_min, axis=0)
    cnt_max = cnt.astype(np.float64)
    cnt_max[cnt == 0] = -1.0
    major_code = np.argmax(cnt_max, axis=0)
    no_allele = ~np.isfinite(cnt_min).any(axis=0)   # SNP sans aucun appel

    gt = ((a1c == minor_code[None, :]).astype(np.float32)
          + (a2c == minor_code[None, :]).astype(np.float32))
    gt[missing] = np.nan
    if no_allele.any():
        gt[:, no_allele] = np.nan

    a1_str = uniq[minor_code].astype(object)
    a2_str = uniq[major_code].astype(object)
    same = (minor_code == major_code)
    a2_str = np.where(same, "0", a2_str)

    ind_df = pd.DataFrame({"FID": fids, "IID": iids})
    allele_df = pd.DataFrame({"A1": a1_str, "A2": a2_str})

    if n_snp_eff < n_snp_map:
        st.info(f"ℹ️ MAP : {n_snp_map} SNPs — {n_snp_eff} chargés, "
                f"les SNPs excédentaires du MAP sont ignorés.")

    return gt, ind_df, allele_df, rejected_short + rejected_empty


# ============================================================
# PARSING VCF
# ============================================================

def parse_vcf(vcf_bytes: bytes) -> tuple:
    """Parse un VCF (gzip auto-détecté), variants bialléliques."""
    text = _decode(vcf_bytes, ".vcf")

    samples = []
    chrs, ids, poss, refs, alts, gt_cols = [], [], [], [], [], []
    n_skipped = n_multi = 0
    gt_cache = {}

    def dosage(gt_str):
        v = gt_cache.get(gt_str)
        if v is not None or gt_str in gt_cache:
            return v
        s = gt_str.replace("|", "/")
        parts = s.split("/")
        try:
            val = float(sum(int(a) for a in parts))
        except ValueError:
            val = np.nan
        gt_cache[gt_str] = val
        return val

    for line in text.splitlines():
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            samples = line.rstrip("\n\r").split("\t")[9:]
            continue
        if line.startswith("#"):
            continue

        parts = line.rstrip("\n\r").split("\t")
        if len(parts) < 10 or not samples:
            n_skipped += 1
            continue
        chrom, pos_s, vid, ref, alt = parts[0], parts[1], parts[2], \
            parts[3], parts[4]
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
        for k, s in enumerate(parts[9:len(samples) + 9]):
            fields = s.split(":")
            if gt_idx >= len(fields):
                continue
            gts[k] = dosage(fields[gt_idx])

        chrs.append(str(chrom))
        ids.append(vid if vid not in (".", "") else f"{chrom}:{pos}")
        poss.append(pos)
        refs.append(ref)
        alts.append(alt)
        gt_cols.append(gts)

    if not gt_cols:
        raise ValueError("Aucun variant biallélique trouvé dans le VCF.")

    gt = np.column_stack(gt_cols).astype(np.float32)
    ref_arr = np.array(refs, dtype=object)
    alt_arr = np.array(alts, dtype=object)

    # Flip pour que le dosage 2 = homozygote allèle MINEUR.
    # [CORRECTION v4] l'inversion est maintenant répercutée sur A1/A2.
    with np.errstate(invalid="ignore"):
        p_alt = np.nanmean(gt, axis=0) / 2.0
    flip = np.isfinite(p_alt) & (p_alt > 0.5)
    if flip.any():
        gt[:, flip] = 2.0 - gt[:, flip]
    a1 = np.where(flip, ref_arr, alt_arr)   # allèle compté (dosage 2)
    a2 = np.where(flip, alt_arr, ref_arr)

    snp_df = pd.DataFrame({
        "CHR": chrs, "SNP": ids, "CM": 0.0, "BP": poss,
        "A1": a1, "A2": a2,
    })
    ind_df = pd.DataFrame({"FID": samples, "IID": samples})
    return gt, ind_df, snp_df, n_skipped, n_multi


# ============================================================
# DONNÉES DE DÉMONSTRATION
# ============================================================

def generate_demo_data(n_ind: int = 150, n_snp: int = 800,
                       n_pop: int = 4, seed: int = 42) -> tuple:
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

    # ROH plausibles chez quelques animaux consanguins
    n_inbred = max(1, n_ind_eff // 15)
    for i in rng.choice(n_ind_eff, n_inbred, replace=False):
        start = int(rng.integers(0, max(1, n_snp - 120)))
        seg = slice(start, start + 120)
        gt[i, seg] = np.where(gt[i, seg] >= 1, 2.0, 0.0)

    mask = rng.random(gt.shape) < 0.02
    gt[mask] = np.nan

    ind_df = pd.DataFrame(ind_rows)
    chr_names = rng.choice(BOVINE_AUTOSOMES, n_snp, p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_UCD12_LENGTHS[c])
                    for c in chr_names], dtype=np.int64)
    snp_df = pd.DataFrame({
        "CHR": chr_names,
        "SNP": [f"rs{i:07d}" for i in range(n_snp)],
        "CM": 0.0,
        "BP": bps,
        "A1": "A",
        "A2": "G",
    })
    snp_df["_k"] = snp_df["CHR"].map(_chr_sort_key)
    order = np.lexsort((snp_df["BP"].values,
                        snp_df["_k"].map(lambda t: t[1]).values))
    snp_df = snp_df.iloc[order].drop(columns="_k").reset_index(drop=True)
    gt = gt[:, order]
    return gt, ind_df, snp_df


# ============================================================
# QC — MÉTRIQUES
# ============================================================

def missingness_per_ind(gt):
    return np.isnan(gt).mean(axis=1)


def missingness_per_snp(gt):
    return np.isnan(gt).mean(axis=0)


def call_rate_per_ind(gt):
    return 1.0 - missingness_per_ind(gt)


def allele_freq(gt):
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt, axis=0) / 2.0


def maf(gt):
    p = allele_freq(gt)
    return np.minimum(p, 1.0 - p)


def heterozygosity(gt):
    """Hétérozygotie observée par individu, sur les génotypes APPELÉS.

    [CORRECTION v4] l'ancienne version divisait par le nombre total de
    SNPs (NaN comptés comme homozygotes) → biais systématique égal au
    taux de données manquantes, qui faussait le filtre het_sd.
    """
    called = ~np.isnan(gt)
    n_called = called.sum(axis=1)
    n_het = (gt == 1).sum(axis=1)
    out = np.full(gt.shape[0], np.nan, dtype=np.float64)
    ok = n_called > 0
    out[ok] = n_het[ok] / n_called[ok]
    return out


def het_expected(gt):
    """Hétérozygotie attendue moyenne par SNP (diversité génique)."""
    p = allele_freq(gt)
    return np.nanmean(2.0 * p * (1.0 - p))


def inbreeding_f_hat(gt):
    """F = 1 - Ho/He par individu (méthode des moments)."""
    ho = heterozygosity(gt)
    he = het_expected(gt)
    if not np.isfinite(he) or he <= 1e-12:
        return np.full_like(ho, np.nan)
    return 1.0 - ho / he


def hwe_exact_p(n_het: int, n_hom1: int, n_hom2: int) -> float:
    """Test exact de Wigginton et al. (2005).

    Validé par énumération exacte. Note : pour des déséquilibres extrêmes
    sur de grands effectifs, la p-value sature autour de 1e-8 (underflow
    flottant) — sans effet sur le filtrage, les seuils usuels étant >= 1e-6.
    """
    n = n_het + n_hom1 + n_hom2
    if n == 0:
        return np.nan
    if n_het == 0 and (n_hom1 == 0 or n_hom2 == 0):
        return 1.0
    rare = 2 * min(n_hom1, n_hom2) + n_het
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2:
        mid += 1
    probs = np.zeros(rare + 1)
    probs[mid] = 1.0
    mysum = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        probs[ch + 2] = probs[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        mysum += probs[ch + 2]
        ch += 2
        chr_ -= 1
        chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        probs[ch - 2] = probs[ch] * ch * (ch - 1) / (4 * (chr_ + 1) * (chc + 1))
        mysum += probs[ch - 2]
        ch -= 2
        chr_ += 1
        chc += 1
    p_obs = probs[n_het] if n_het < len(probs) else 0.0
    return float(min(probs[probs <= p_obs + 1e-7].sum() / mysum, 1.0))


@cache_data
def hwe_pvalues(gt: np.ndarray) -> np.ndarray:
    """Test HWE : exact (Wigginton) si peu de SNPs, sinon chi² corrigé."""
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
        denom = np.where(n_valid > 0, n_valid, np.nan)
        p = (n1 + 2 * n2) / (2.0 * denom)
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
            f"⚠️ Alignement corrigé — gt={gt.shape}, ind_df={len(ind_df)}, "
            f"snp_df={len(snp_df)} → ({n_ind}, {n_snp})")
    return (gt[:n_ind, :n_snp],
            ind_df.iloc[:n_ind].reset_index(drop=True),
            snp_df.iloc[:n_snp].reset_index(drop=True))


def apply_qc_filters(gt, ind_df, snp_df, params):
    """QC séquentiel : geno → mind → MAF → HWE → hétérozygotie."""
    gt, ind_df, snp_df = _align_shapes(gt, ind_df, snp_df)
    n0, m0 = gt.shape
    log = []

    keep_snp = missingness_per_snp(gt) <= params["geno"]
    log.append(("Missingness SNP (geno)", int((~keep_snp).sum()), "SNP"))
    gt = gt[:, keep_snp]
    snp_df = snp_df[keep_snp].reset_index(drop=True)
    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par le filtre de missingness.")

    keep_ind = missingness_per_ind(gt) <= params["mind"]
    log.append(("Missingness individu (mind)", int((~keep_ind).sum()), "Ind"))
    gt = gt[keep_ind]
    ind_df = ind_df[keep_ind].reset_index(drop=True)
    if gt.shape[0] == 0:
        raise ValueError("Tous les individus exclus (missingness).")

    m = maf(gt)
    keep_maf = np.isfinite(m) & (m >= params["maf"])
    log.append(("MAF", int((~keep_maf).sum()), "SNP"))
    gt = gt[:, keep_maf]
    snp_df = snp_df[keep_maf].reset_index(drop=True)
    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par MAF.")

    pv = hwe_pvalues(gt)
    keep_hwe = np.isnan(pv) | (pv >= params["hwe"])
    log.append(("HWE", int((~keep_hwe).sum()), "SNP"))
    gt = gt[:, keep_hwe]
    snp_df = snp_df[keep_hwe].reset_index(drop=True)
    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par HWE.")

    het = heterozygosity(gt)
    sd = np.nanstd(het)
    if np.isfinite(sd) and sd > 1e-9:
        z = (het - np.nanmean(het)) / sd
        keep_het = np.isfinite(z) & (np.abs(z) <= params["het_sd"])
    else:
        keep_het = np.ones_like(het, dtype=bool)
    log.append(("Hétérozygotie (outliers)", int((~keep_het).sum()), "Ind"))
    gt = gt[keep_het]
    ind_df = ind_df[keep_het].reset_index(drop=True)
    if gt.shape[0] == 0:
        raise ValueError("Tous les individus exclus par hétérozygotie.")

    qc_stats = {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]),
        "excluded_snp": int(m0 - gt.shape[1]),
        "call_rate": float(1.0 - np.isnan(gt).mean()),
        "het_obs": float(np.nanmean(heterozygosity(gt))),
        "het_exp": float(het_expected(gt)),
        "maf_mean": float(np.nanmean(maf(gt))),
        "log": log,
    }
    return gt, ind_df, snp_df, qc_stats


# ============================================================
# GÉNÉTIQUE DES POPULATIONS
# ============================================================

@cache_data
def fst_per_snp(gt: np.ndarray, pop_labels: np.ndarray,
                min_n: int = 3) -> np.ndarray:
    """FST de Nei par SNP (vectorisé)."""
    pop_labels = as_str_array(pop_labels)
    pops = np.unique(pop_labels)
    n_snp = gt.shape[1]
    if len(pops) < 2:
        return np.full(n_snp, np.nan)

    P = np.empty((len(pops), n_snp), dtype=np.float64)
    N = np.empty((len(pops), n_snp), dtype=np.float64)
    for i, p in enumerate(pops):
        g = gt[pop_labels == p]
        called = ~np.isnan(g)
        n = called.sum(axis=0)
        with np.errstate(invalid="ignore"):
            P[i] = np.nanmean(g, axis=0) / 2.0
        N[i] = n

    ok = (N >= min_n) & np.isfinite(P)
    W = np.where(ok, N, 0.0)
    Pm = np.where(ok, P, 0.0)
    tot = W.sum(axis=0)
    n_ok = ok.sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        p_bar = (Pm * W).sum(axis=0) / tot
        h_s = (2.0 * Pm * (1.0 - Pm) * W).sum(axis=0) / tot
        h_t = 2.0 * p_bar * (1.0 - p_bar)
        fst = (h_t - h_s) / h_t
    fst[(n_ok < 2) | (tot <= 0) | ~np.isfinite(h_t) | (h_t <= 1e-9)] = np.nan
    return fst


@cache_data
def fst_pairwise(gt: np.ndarray, pop_labels: np.ndarray) -> tuple:
    """Matrice FST par paires de populations."""
    pop_labels = as_str_array(pop_labels)
    pops = sorted(np.unique(pop_labels).tolist())
    K = len(pops)
    matrix = np.full((K, K), np.nan)
    np.fill_diagonal(matrix, 0.0)
    if K < 2:
        return matrix, pops

    freqs, ns = {}, {}
    for p in pops:
        g = gt[pop_labels == p]
        with np.errstate(invalid="ignore"):
            freqs[p] = np.nanmean(g, axis=0) / 2.0
        ns[p] = (~np.isnan(g)).sum(axis=0).astype(float)

    for i in range(K):
        for j in range(i + 1, K):
            p1, p2 = freqs[pops[i]], freqs[pops[j]]
            n1, n2 = ns[pops[i]], ns[pops[j]]
            ok = (n1 > 0) & (n2 > 0) & np.isfinite(p1) & np.isfinite(p2)
            denom = np.where(ok, n1 + n2, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                p_bar = (p1 * n1 + p2 * n2) / denom
                hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / denom
                ht = 2.0 * p_bar * (1.0 - p_bar)
                fst_j = np.where(ht > 1e-9, (ht - hs) / ht, np.nan)
            val = float(np.nanmean(fst_j)) if np.isfinite(fst_j).any() else np.nan
            matrix[i, j] = matrix[j, i] = val
    return matrix, pops


@cache_data
def pca_analysis(gt: np.ndarray, n_components: int = 10) -> tuple:
    X = impute_mean(gt)
    X = X - X.mean(axis=0)
    n_comp = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
    scores = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100.0
    if scores.shape[1] < 2:      # garantit PC1/PC2 pour les tracés
        scores = np.column_stack([scores, np.zeros(len(scores))])
        var = np.append(var, 0.0)
    return scores, var


def ibs_distance(gt: np.ndarray, max_snp: int = MDS_MAX_SNP,
                 seed: int = 42) -> np.ndarray:
    """Distance de partage allélique (1 - IBS), via produits matriciels.

    |xi - xj| se décompose sur les indicatrices de génotype, ce qui permet
    de remplacer la double boucle O(n²m) par 3 produits BLAS.
    """
    n_snp = gt.shape[1]
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n_snp, max_snp, replace=False))
           if n_snp > max_snp else np.arange(n_snp))
    X = np.clip(np.rint(impute_mean(gt[:, idx])), 0, 2)
    A0 = (X == 0).astype(np.float32)
    A1 = (X == 1).astype(np.float32)
    A2 = (X == 2).astype(np.float32)
    m = float(X.shape[1])
    d1 = A0 @ A1.T
    d1 = d1 + d1.T
    d1b = A1 @ A2.T
    d1 = d1 + d1b + d1b.T
    d2 = A0 @ A2.T
    d2 = d2 + d2.T
    D = (d1 + 2.0 * d2) / (2.0 * m)   # ∈ [0, 1]
    np.fill_diagonal(D, 0.0)
    return np.asarray(D, dtype=np.float64)


@cache_data
def mds_analysis(gt: np.ndarray, n_components: int = 5) -> np.ndarray:
    """MDS métrique sur la distance IBS.

    Les paramètres sont construits dynamiquement : scikit-learn a renommé
    `dissimilarity` en `metric` et modifié le défaut de `init`.
    """
    D = ibs_distance(gt)
    n_comp = max(2, min(n_components, D.shape[0] - 1))
    sig = set(inspect.signature(SklearnMDS.__init__).parameters)
    kw = {"n_components": n_comp, "random_state": 42,
          "n_init": 1, "max_iter": 300}
    if "metric" in sig and "dissimilarity" not in sig:
        kw["metric"] = "precomputed"
    else:
        kw["dissimilarity"] = "precomputed"
    if "normalized_stress" in sig:
        kw["normalized_stress"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SklearnMDS(**kw).fit_transform(D)


@cache_data
def ld_decay(gt: np.ndarray, snp_chr: np.ndarray, snp_bp: np.ndarray,
             max_kb: float = 1000, max_snp: int = 1500,
             seed: int = 42) -> pd.DataFrame:
    """r² intra-chromosomique en fonction de la distance physique."""
    n_snp = gt.shape[1]
    if n_snp < 2:
        return pd.DataFrame(columns=["dist_kb", "r2"])
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n_snp, max_snp, replace=False))
           if n_snp > max_snp else np.arange(n_snp))

    chr_sub = as_str_array(snp_chr)[idx]
    bp_sub = np.asarray(snp_bp, dtype=np.float64)[idx]
    X = impute_mean(gt[:, idx])
    X = X - X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = np.nan
    X = X / sd

    dists, r2s = [], []
    for chrom in pd.unique(chr_sub):
        sel = np.where(chr_sub == chrom)[0]
        if len(sel) < 2:
            continue
        Xi = X[:, sel]
        C = (Xi.T @ Xi) / Xi.shape[0]
        R2 = C ** 2
        iu, ju = np.triu_indices(len(sel), k=1)
        d = np.abs(bp_sub[sel][ju] - bp_sub[sel][iu]) / 1000.0
        mask = (d > 0) & (d <= max_kb) & np.isfinite(R2[iu, ju])
        dists.append(d[mask])
        r2s.append(R2[iu[mask], ju[mask]])

    if not dists:
        return pd.DataFrame(columns=["dist_kb", "r2"])
    return pd.DataFrame({"dist_kb": np.concatenate(dists),
                         "r2": np.concatenate(r2s)})


@cache_data
def kinship_matrix(gt: np.ndarray) -> np.ndarray:
    """GRM VanRaden (méthode 2, marqueurs standardisés)."""
    X = impute_mean(gt)
    p = np.clip(X.mean(axis=0) / 2.0, 1e-3, 1 - 1e-3)
    Z = X - 2 * p
    Zn = Z / np.sqrt(2 * p * (1 - p))
    return (Zn @ Zn.T) / X.shape[1]


@cache_data
def admixture_nmf(gt: np.ndarray, K: int = 3, seed: int = 42,
                  max_iter: int = 500) -> tuple:
    """Décomposition NMF des dosages (approximation d'ADMIXTURE)."""
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    model = NMF(n_components=K, init="nndsvda", random_state=seed,
                max_iter=max_iter)
    W = model.fit_transform(X)
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return W / s, model.components_


# ============================================================
# ROH — VERSION VECTORISÉE
# ============================================================

@cache_data
def detect_roh(gt: np.ndarray, chr_arr: np.ndarray, bp: np.ndarray,
               min_snps: int = 30, min_kb: float = 500.0) -> tuple:
    """Runs of homozygosity par encodage RLE vectorisé.

    [CORRECTION v4] plus de sérialisation JSON du snp_df (dépréciée et
    source de désalignement) : on reçoit directement les tableaux.
    """
    chr_arr = as_str_array(chr_arr)
    bp = np.asarray(bp, dtype=np.int64)
    n_ind = gt.shape[0]
    froh_bp = np.zeros(n_ind, dtype=np.float64)
    genome_bp_total = 0.0
    chunks = []

    for chrom in pd.unique(chr_arr):
        idxs = np.where(chr_arr == chrom)[0]
        if len(idxs) < min_snps:
            continue
        idxs = idxs[np.argsort(bp[idxs], kind="stable")]
        cbp = bp[idxs]
        span = float(cbp[-1] - cbp[0])
        if span < 1:
            continue
        genome_bp_total += span

        hom = (gt[:, idxs] == 0) | (gt[:, idxs] == 2)
        pad = np.zeros((n_ind, 1), dtype=np.int8)
        m = np.concatenate([pad, hom.astype(np.int8), pad], axis=1)
        d = np.diff(m, axis=1)
        si = np.argwhere(d == 1)
        ei = np.argwhere(d == -1)
        if si.size == 0:
            continue
        rows = si[:, 0]
        s_idx = si[:, 1]
        e_idx = ei[:, 1] - 1
        n_run = e_idx - s_idx + 1
        length_bp = cbp[e_idx] - cbp[s_idx]
        length_kb = length_bp / 1000.0
        keep = (n_run >= min_snps) & (length_kb >= min_kb)
        if not keep.any():
            continue
        rows, s_idx, e_idx = rows[keep], s_idx[keep], e_idx[keep]
        np.add.at(froh_bp, rows, length_bp[keep].astype(np.float64))
        chunks.append(pd.DataFrame({
            "IID_idx": rows,
            "CHR": chrom,
            "start_bp": cbp[s_idx],
            "end_bp": cbp[e_idx],
            "n_snp": n_run[keep],
            "length_kb": length_kb[keep],
        }))

    froh = (froh_bp / genome_bp_total if genome_bp_total > 0
            else np.zeros(n_ind))
    if chunks:
        roh_df = pd.concat(chunks, ignore_index=True)
    else:
        roh_df = pd.DataFrame(columns=["IID_idx", "CHR", "start_bp",
                                       "end_bp", "n_snp", "length_kb"])
    return roh_df, froh


# ============================================================
# EXPORT PLINK / VCF
# ============================================================

def build_plink_bed(gt: np.ndarray) -> bytes:
    """Écrit un .bed SNP-major (A1 = allèle compté du snp_df)."""
    n_ind, n_snp = gt.shape
    pad = (-n_ind) % 4
    out = [bytes([0x6C, 0x1B, 0x01])]
    for j in range(n_snp):
        col = gt[:, j]
        bits = np.full(n_ind, 0b01, dtype=np.uint8)   # 01 = manquant
        bits[col == 2] = 0b00                          # A1/A1
        bits[col == 1] = 0b10                          # hétérozygote
        bits[col == 0] = 0b11                          # A2/A2
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        b = bits.reshape(-1, 4)
        packed = (b[:, 0] | (b[:, 1] << 2) | (b[:, 2] << 4)
                  | (b[:, 3] << 6)).astype(np.uint8)
        out.append(packed.tobytes())
    return b"".join(out)


def build_plink_bim(snp_df: pd.DataFrame) -> str:
    df = _ensure_alleles(snp_df)
    lines = [
        f"{c}\t{s}\t{cm}\t{int(bp)}\t{a1}\t{a2}"
        for c, s, cm, bp, a1, a2 in zip(
            df["CHR"].astype(str), df["SNP"].astype(str),
            df["CM"] if "CM" in df.columns else [0] * len(df),
            df["BP"].astype(np.int64), df["A1"], df["A2"])
    ]
    return "\n".join(lines) + "\n"


def build_plink_fam(ind_df: pd.DataFrame) -> str:
    lines = [f"{f}\t{i}\t0\t0\t0\t-9"
             for f, i in zip(ind_df["FID"].astype(str),
                             ind_df["IID"].astype(str))]
    return "\n".join(lines) + "\n"


def build_vcf_output(gt: np.ndarray, ind_df: pd.DataFrame,
                     snp_df: pd.DataFrame, project: str = "BovineSNP") -> str:
    """VCF 4.2. Le dosage 2 (= A1/A1) devient 1/1, donc ALT = A1."""
    df = _ensure_alleles(snp_df)
    n_ind, n_snp = gt.shape
    samples = ind_df["IID"].astype(str).tolist()
    header = [
        "##fileformat=VCFv4.2",
        f"##source=BovineSNPPlatform-{APP_VERSION}-{project}",
        f"##fileDate={datetime.now():%Y%m%d}",
        "##reference=ARS-UCD1.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples),
    ]
    chrs = df["CHR"].astype(str).values
    bps = df["BP"].astype(np.int64).values
    ids = df["SNP"].astype(str).values
    a1 = df["A1"].values
    a2 = df["A2"].values

    body = []
    for j in range(n_snp):
        col = gt[:, j]
        calls = np.where(np.isnan(col), "./.",
                         np.where(col == 0, "0/0",
                                  np.where(col == 1, "0/1", "1/1")))
        body.append(f"{chrs[j]}\t{bps[j]}\t{ids[j]}\t{a2[j]}\t{a1[j]}"
                    f"\t.\tPASS\t.\tGT\t" + "\t".join(calls.tolist()))
    return "\n".join(header + body) + "\n"


def build_plink_zip(gt, ind_df, snp_df, prefix="bovine_qc") -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", build_plink_bed(gt))
        zf.writestr(f"{prefix}.bim", build_plink_bim(snp_df))
        zf.writestr(f"{prefix}.fam", build_plink_fam(ind_df))
    return bio.getvalue()


# ============================================================
# VISUALISATION
# ============================================================

_LAYOUT = dict(margin=dict(l=45, r=20, t=60, b=45))


def plot_hist(values, title, xlabel, color="#3498db"):
    fig = go.Figure(go.Histogram(x=np.asarray(values), nbinsx=80,
                                 marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Fréquence", height=380, **_LAYOUT)
    return fig


def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Missingness / individu",
                                        "Missingness / SNP"))
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60,
                               marker_color="skyblue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60,
                               marker_color="coral"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False, **_LAYOUT)
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=1)
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=2)
    fig.update_yaxes(title_text="Nombre", row=1, col=1)
    return fig


def plot_pca(scores, var_pct, labels, pc_x=0, pc_y=1):
    df = pd.DataFrame({
        f"PC{pc_x+1}": scores[:, pc_x],
        f"PC{pc_y+1}": scores[:, pc_y],
        "Population": as_str_array(labels)})
    fig = px.scatter(
        df, x=f"PC{pc_x+1}", y=f"PC{pc_y+1}", color="Population",
        title=(f"ACP — PC{pc_x+1} ({var_pct[pc_x]:.1f}%) vs "
               f"PC{pc_y+1} ({var_pct[pc_y]:.1f}%)"), height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(**_LAYOUT)
    return fig


def plot_scree(var_pct):
    n = min(10, len(var_pct))
    fig = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(n)],
                           y=var_pct[:n], marker_color="#34495e"))
    fig.update_layout(title="Variance expliquée par composante",
                      yaxis_title="%", height=330, **_LAYOUT)
    return fig


def plot_mds(coords, labels):
    df = pd.DataFrame({"MDS1": coords[:, 0], "MDS2": coords[:, 1],
                       "Population": as_str_array(labels)})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population",
                     title="MDS (distance IBS) — structure des populations",
                     height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(**_LAYOUT)
    return fig


def plot_manhattan(fst, chr_col, bp_col=None, threshold_q=0.999):
    """Manhattan FST, positions cumulées en bp quand elles sont fournies."""
    fst = np.asarray(fst, dtype=float)
    chrs = as_str_array(chr_col)
    bps = (np.asarray(bp_col, dtype=np.float64) if bp_col is not None
           else np.arange(len(fst), dtype=np.float64))
    ok = np.isfinite(fst)
    if not ok.any():
        return None
    fst, chrs, bps = fst[ok], chrs[ok], bps[ok]

    keys = np.array([_chr_sort_key(c)[1] for c in chrs], dtype=float)
    order = np.lexsort((bps, keys))
    fst, chrs, bps = fst[order], chrs[order], bps[order]

    x = np.empty_like(bps)
    offset, ticks, labels = 0.0, [], []
    for chrom in pd.unique(chrs):
        sel = chrs == chrom
        local = bps[sel] - bps[sel].min()
        x[sel] = local + offset
        ticks.append(offset + local.max() / 2.0)
        labels.append(str(chrom))
        offset += max(local.max(), 1.0) * 1.02
    q_upper = float(np.nanquantile(fst, threshold_q))

    fig = go.Figure()
    for i, chrom in enumerate(pd.unique(chrs)):
        sel = chrs == chrom
        fig.add_trace(go.Scatter(
            x=x[sel], y=fst[sel], mode="markers",
            marker=dict(size=5,
                        color="#2c3e50" if i % 2 == 0 else "#7f8c8d"),
            name=f"chr{chrom}", showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=q_upper, line_dash="dash", line_color="red",
                  annotation_text=(f"Top {100*(1-threshold_q):.2f}% = "
                                   f"{q_upper:.4f}"),
                  annotation_position="top right")
    fig.update_layout(title="Manhattan — FST par SNP",
                      xaxis_title="Chromosome", yaxis_title="FST",
                      height=500, **_LAYOUT)
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_ld_decay(ld_df, bin_kb=20):
    if ld_df is None or ld_df.empty:
        return None
    d = ld_df.copy()
    d["bin"] = (d["dist_kb"] // bin_kb) * bin_kb + bin_kb / 2.0
    agg = d.groupby("bin")["r2"].agg(["mean", "count"]).reset_index()
    agg = agg[agg["count"] >= 5]
    if agg.empty:
        return None
    fig = px.line(agg, x="bin", y="mean",
                  labels={"bin": "Distance (kb)", "mean": "r² moyen"},
                  title="Déséquilibre de liaison (LD decay, intra-chromosome)",
                  height=450)
    fig.update_traces(line=dict(color="royalblue", width=3))
    fig.update_layout(**_LAYOUT)
    return fig


def plot_kinship_heatmap(G, labels):
    labels = [str(x) for x in labels]
    n = len(labels)
    seen, uniq = {}, []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        uniq.append(lab if seen[lab] == 1 else f"{lab}#{seen[lab]}")
    vmin = float(np.nanpercentile(G, 1))
    vmax = float(np.nanpercentile(G, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -0.3, 0.5
    custom = np.stack([np.repeat(uniq, n).reshape(n, n),
                       np.tile(uniq, n).reshape(n, n)], axis=-1)
    fig = go.Figure(go.Heatmap(
        z=G, colorscale="RdBu", zmid=0.0, zmin=vmin, zmax=vmax,
        colorbar=dict(title="GRM"),
        hovertemplate=("%{customdata[0]} × %{customdata[1]}"
                       "<br>Parenté = %{z:.3f}<extra></extra>"),
        customdata=custom))
    fig.update_layout(title="Matrice de parenté génomique (GRM)", height=650,
                      xaxis=dict(title="Individus", showticklabels=False),
                      yaxis=dict(title="Individus", showticklabels=False,
                                 autorange="reversed"), **_LAYOUT)
    return fig


def plot_admixture(Q, labels, pop_labels, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
    df["IID"] = as_str_array(labels)
    df["Pop"] = as_str_array(pop_labels)
    df["_dom"] = Q.argmax(axis=1)
    df["_maxq"] = -Q.max(axis=1)
    df["_pop_order"] = pd.Categorical(df["Pop"],
                                      categories=sorted(set(df["Pop"])))
    df = df.sort_values(["_pop_order", "_dom", "_maxq"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))

    palette = (px.colors.qualitative.Set2 + px.colors.qualitative.Set3
               + px.colors.qualitative.Dark24)
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(
            x=df["x"], y=df[f"K{k+1}"], name=f"Composante {k+1}",
            marker_color=palette[k % len(palette)],
            marker_line_width=0,
            hovertemplate="%{customdata}<br>K" + str(k + 1)
                          + " = %{y:.3f}<extra></extra>",
            customdata=df["IID"]))
    # séparateurs entre populations
    bounds = np.where(df["Pop"].values[1:] != df["Pop"].values[:-1])[0]
    for b in bounds:
        fig.add_vline(x=b + 0.5, line_width=1, line_color="black")
    fig.update_layout(barmode="stack", bargap=0.0,
                      title=f"Proportions d'ancestralité — NMF (K={K})",
                      xaxis_title="Individus (triés par population)",
                      yaxis_title="Proportion", height=500, **_LAYOUT)
    return fig, df


def plot_fst_pairwise(matrix, pops):
    fig = go.Figure(go.Heatmap(
        z=matrix, x=pops, y=pops, colorscale="Viridis",
        colorbar=dict(title="FST"),
        text=np.round(matrix, 4), texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>FST = %{z:.4f}<extra></extra>"))
    fig.update_layout(title="FST par paires de populations", height=550,
                      **_LAYOUT)
    return fig


def plot_roh_histogram(froh, labels):
    df = pd.DataFrame({"FROH": froh,
                       "Population": as_str_array(labels)})
    fig = px.histogram(df, x="FROH", color="Population", nbins=50,
                       title="Distribution de FROH (fraction du génome en ROH)",
                       height=450)
    fig.update_layout(**_LAYOUT)
    return fig


def plot_roh_length_classes(roh_df):
    if roh_df is None or roh_df.empty:
        return None
    bins = [0, 1_000, 2_000, 4_000, 8_000, 16_000, np.inf]
    names = ["<1 Mb", "1–2 Mb", "2–4 Mb", "4–8 Mb", "8–16 Mb", ">16 Mb"]
    cat = pd.cut(roh_df["length_kb"], bins=bins, labels=names, right=False)
    agg = cat.value_counts().reindex(names).fillna(0).reset_index()
    agg.columns = ["Classe", "Nombre"]
    fig = px.bar(agg, x="Classe", y="Nombre",
                 title="Classes de longueur des ROH "
                       "(proxy de l'ancienneté de la consanguinité)",
                 height=400)
    fig.update_traces(marker_color="#c0392b")
    fig.update_layout(**_LAYOUT)
    return fig


def plot_roh_manhattan(roh_df, ind_labels, max_ind_display=50):
    """Carte des ROH — une seule trace Plotly (segments séparés par None)."""
    if roh_df is None or roh_df.empty:
        return None
    df = roh_df.copy()
    df["IID"] = [str(ind_labels[i]) for i in df["IID_idx"]]
    if df["IID"].nunique() > max_ind_display:
        keep = df["IID"].value_counts().head(max_ind_display).index
        df = df[df["IID"].isin(keep)]
    df = df.sort_values(["CHR", "start_bp"])

    chroms = sorted(df["CHR"].astype(str).unique(), key=_chr_sort_key)
    chr_offset, offset, ticks = {}, 0, []
    for chrom in chroms:
        chr_len = ARS_UCD12_LENGTHS.get(str(chrom).upper(), 100_000_000)
        chr_offset[chrom] = offset
        ticks.append(offset + chr_len / 2)
        offset += chr_len
    iid_list = sorted(df["IID"].unique())
    y_map = {iid: i for i, iid in enumerate(iid_list)}

    xs, ys, hov = [], [], []
    for r in df.itertuples(index=False):
        off = chr_offset[str(r.CHR)]
        y = y_map[r.IID]
        xs += [off + r.start_bp, off + r.end_bp, None]
        ys += [y, y, None]
        txt = (f"{r.IID} — chr{r.CHR}<br>{r.length_kb:,.0f} kb "
               f"({r.n_snp} SNPs)")
        hov += [txt, txt, ""]

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="crimson", width=5),
        hovertext=hov, hoverinfo="text", showlegend=False,
        connectgaps=False))
    fig.update_layout(title="Carte des ROH par individu",
                      xaxis_title="Position génomique cumulée",
                      yaxis_title="Individu",
                      height=max(400, 16 * len(iid_list)), **_LAYOUT)
    fig.update_xaxes(tickvals=ticks, ticktext=chroms)
    fig.update_yaxes(tickvals=list(y_map.values()),
                     ticktext=list(y_map.keys()),
                     tickfont=dict(size=9))
    return fig


# ============================================================
# RAPPORT HTML
# ============================================================

def build_report_html(config, qc_stats, figures=None, tables=None):
    figures = figures or {}
    tables = tables or {}
    first, blocks = True, []
    for title, fig in figures.items():
        if fig is None:
            continue
        html = fig.to_html(full_html=False,
                           include_plotlyjs="cdn" if first else False)
        first = False
        blocks.append(f"<h2>{title}</h2>{html}")
    for title, df in tables.items():
        if df is None or len(df) == 0:
            continue
        blocks.append(f"<h2>{title}</h2>{df.to_html(index=False, border=0)}")
    figs_html = "\n".join(blocks) if blocks else "<p>Aucune figure.</p>"

    log_rows = "".join(
        f"<tr><td>{name}</td><td>{n}</td><td>{unit}</td></tr>"
        for name, n, unit in qc_stats.get("log", []))

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Rapport Bovine SNP Platform</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px;
        color: #222; }}
 h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 8px; }}
 h2 {{ color: #2471a3; margin-top: 28px; }}
 .summary {{ background: #f4f6f7; padding: 16px; border-radius: 8px; }}
 table {{ border-collapse: collapse; margin-top: 10px; }}
 th, td {{ border: 1px solid #ccc; padding: 5px 12px; font-size: 0.9em; }}
 th {{ background: #eaf2f8; }}
</style></head><body>
<h1>🐄 Rapport Bovine SNP Platform</h1>
<p><b>Projet :</b> {config.get('project_name', 'N/A')} —
   <b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="summary"><h2>Résumé exécutif</h2><ul>
 <li>Individus analysés : <b>{qc_stats['n_ind_final']}</b>
     (sur {qc_stats['n_ind_init']})</li>
 <li>SNPs retenus : <b>{qc_stats['n_snp_final']}</b>
     (sur {qc_stats['n_snp_init']})</li>
 <li>Individus exclus (QC) : <b>{qc_stats['excluded_ind']}</b></li>
 <li>SNPs exclus (QC) : <b>{qc_stats['excluded_snp']}</b></li>
 <li>Taux d'appel final : <b>{qc_stats.get('call_rate', float('nan')):.4f}</b></li>
 <li>Hétérozygotie observée / attendue :
     <b>{qc_stats.get('het_obs', float('nan')):.4f}</b> /
     <b>{qc_stats.get('het_exp', float('nan')):.4f}</b></li>
 <li>MAF moyenne : <b>{qc_stats.get('maf_mean', float('nan')):.4f}</b></li>
</ul>
<h2>Détail des exclusions</h2>
<table><tr><th>Filtre</th><th>Exclus</th><th>Unité</th></tr>
{log_rows}</table>
</div>
{figs_html}
<hr><p style="font-size:0.85em; color:#666">
Rapport généré automatiquement par Bovine SNP Platform v{APP_VERSION}.
Assemblage de référence : ARS-UCD1.2.</p>
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
    "admixture_Q", "admixture_K",
    "roh_df", "froh", "run_requested",
]

_DERIVED_KEYS = ["pca_scores", "pca_var", "mds_coords", "kinship", "ld_df",
                 "fst", "fst_pairwise_matrix", "fst_pops", "admixture_Q",
                 "admixture_K", "roh_df", "froh"]


def render_html(html: str, height: int = 700):
    """Aperçu HTML — st.iframe (récent) avec repli sur components.v1.html."""
    try:
        st.components.v1.html(html, height=height, scrolling=True)
    except Exception:
        # st.iframe n'accepte qu'une URL, pas du HTML inline : on se
        # contente alors d'inviter au téléchargement.
        st.info("Prévisualisation indisponible dans cette version de "
                "Streamlit — téléchargez le rapport pour l'ouvrir.")


def init_state():
    for k in _STATE_KEYS:
        st.session_state.setdefault(k, None)


def invalidate_downstream():
    for k in _DERIVED_KEYS:
        st.session_state[k] = None


def reset_all_derived():
    for k in ["gt_filt", "ind_filt", "snp_filt", "qc_stats"] + _DERIVED_KEYS:
        st.session_state[k] = None
    st.session_state.pop("_plink_zip", None)
    st.session_state.pop("_vcf_str", None)


def has_data() -> bool:
    return st.session_state.get("gt") is not None


def has_qc() -> bool:
    return (st.session_state.get("gt_filt") is not None
            and st.session_state.get("qc_stats") is not None
            and st.session_state.get("ind_filt") is not None
            and st.session_state.get("snp_filt") is not None)


def run_full_pipeline(params, K=4, roh_min_snps=30, roh_min_kb=500.0):
    """Exécute QC + toutes les analyses en une passe."""
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
        gt_f, as_str_array(snp_f["CHR"]),
        as_int_array(snp_f["BP"]), max_kb=1000, max_snp=1500)
    labels = as_str_array(ind_f["FID"])
    st.session_state.fst = fst_per_snp(gt_f, labels)
    mat, pops = fst_pairwise(gt_f, labels)
    st.session_state.fst_pairwise_matrix = mat
    st.session_state.fst_pops = pops
    K = int(min(K, max(2, gt_f.shape[0] - 1)))
    Q, _ = admixture_nmf(gt_f, K=K)
    st.session_state.admixture_Q = Q
    st.session_state.admixture_K = K
    roh_df, froh = detect_roh(
        gt_f, as_str_array(snp_f["CHR"]),
        as_int_array(snp_f["BP"]),
        min_snps=int(roh_min_snps), min_kb=float(roh_min_kb))
    st.session_state.roh_df = roh_df
    st.session_state.froh = froh


def collect_figures():
    """Reconstruit les figures disponibles pour le rapport."""
    figures, tables = {}, {}
    ss = st.session_state
    if not has_qc():
        return figures, tables
    pops = as_str_array(ss.ind_filt["FID"])
    iids = as_str_array(ss.ind_filt["IID"])

    figures["QC — missingness"] = plot_missingness_dashboard(
        missingness_per_ind(ss.gt_filt), missingness_per_snp(ss.gt_filt))
    figures["QC — spectre MAF"] = plot_hist(
        maf(ss.gt_filt), "Spectre MAF (post-QC)", "MAF", "#2ecc71")
    if ss.pca_scores is not None:
        figures["ACP"] = plot_pca(ss.pca_scores, ss.pca_var, pops)
        figures["ACP — variance"] = plot_scree(ss.pca_var)
    if ss.mds_coords is not None:
        figures["MDS (IBS)"] = plot_mds(ss.mds_coords, pops)
    if ss.kinship is not None:
        figures["GRM"] = plot_kinship_heatmap(
            ss.kinship, [f"{p}_{i}" for p, i in zip(pops, iids)])
    if ss.admixture_Q is not None:
        fig_adm, df_adm = plot_admixture(ss.admixture_Q, iids, pops,
                                         ss.admixture_K)
        figures["Admixture (NMF)"] = fig_adm
        tables["Ancestralité moyenne par population"] = (
            df_adm.groupby("Pop")[[f"K{k+1}" for k in
                                   range(ss.admixture_K)]]
            .mean().round(3).reset_index())
    if ss.ld_df is not None and not ss.ld_df.empty:
        figures["LD decay"] = plot_ld_decay(ss.ld_df)
    if ss.fst is not None:
        figures["Manhattan FST"] = plot_manhattan(
            ss.fst, as_str_array(ss.snp_filt["CHR"]),
            as_int_array(ss.snp_filt["BP"]))
    if ss.fst_pairwise_matrix is not None:
        figures["FST pairwise"] = plot_fst_pairwise(
            ss.fst_pairwise_matrix, ss.fst_pops)
    if ss.froh is not None:
        figures["FROH"] = plot_roh_histogram(ss.froh, pops)
        figures["Classes de ROH"] = plot_roh_length_classes(ss.roh_df)
    return figures, tables


# ============================================================
# STREAMLIT — INTERFACE
# ============================================================

def _sidebar_data_loader():
    st.header("📁 Données")
    mode = st.radio("Source :", ["Démo", "Upload PED/MAP", "Upload VCF"],
                    index=0)

    if mode == "Démo":
        c1, c2 = st.columns(2)
        n_ind = c1.number_input("Individus", 20, 2000, 150, 10)
        n_snp = c2.number_input("SNPs", 100, 50000, 2000, 100)
        n_pop = st.slider("Populations", 2, 10, 4)
        if st.button("🎲 Générer le jeu de démo", **STRETCH):
            with st.spinner("Génération..."):
                gt, ind_df, snp_df = generate_demo_data(
                    int(n_ind), int(n_snp), int(n_pop))
            st.session_state.gt = gt
            st.session_state.ind_df = ind_df
            st.session_state.snp_df = snp_df
            reset_all_derived()
            st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")

    elif mode == "Upload PED/MAP":
        st.caption("Formats : .ped / .ped.gz + .map / .map.gz")
        ped_file = st.file_uploader("Fichier .ped", type=["ped", "gz", "txt"])
        map_file = st.file_uploader("Fichier .map", type=["map", "gz", "txt"])
        if ped_file and map_file and st.button("📥 Charger PED + MAP",
                                               **STRETCH):
            try:
                with st.spinner("Lecture du .map..."):
                    map_df, rej_map = parse_map(map_file.read())
                st.info(f"📋 MAP : **{len(map_df)}** SNPs")
                with st.spinner(f"Lecture du .ped "
                                f"({ped_file.size/1024/1024:.1f} Mo)..."):
                    gt, ind_df, allele_df, rej_ped = parse_ped(
                        ped_file.read(), len(map_df))
                map_df = map_df.iloc[:gt.shape[1]].reset_index(drop=True)
                map_df["A1"] = allele_df["A1"].values[:len(map_df)]
                map_df["A2"] = allele_df["A2"].values[:len(map_df)]
                st.session_state.gt = gt
                st.session_state.ind_df = ind_df
                st.session_state.snp_df = map_df
                reset_all_derived()
                st.success(f"✅ **{gt.shape[0]}** individus × "
                           f"**{gt.shape[1]}** SNPs")
                if rej_map or rej_ped:
                    st.warning(f"⚠️ Lignes rejetées : {rej_map} (map), "
                               f"{rej_ped} (ped)")
            except Exception as exc:
                st.error(f"❌ {exc}")

    else:
        vcf_file = st.file_uploader("Fichier .vcf / .vcf.gz",
                                    type=["vcf", "gz", "txt"])
        if vcf_file is not None and st.button("📥 Charger le VCF",
                                              **STRETCH):
            try:
                with st.spinner("Lecture du VCF..."):
                    gt, ind_df, snp_df, n_skip, n_multi = parse_vcf(
                        vcf_file.read())
                st.session_state.gt = gt
                st.session_state.ind_df = ind_df
                st.session_state.snp_df = snp_df
                reset_all_derived()
                msg = f"✅ {gt.shape[0]} ind × {gt.shape[1]} variants"
                if n_skip or n_multi:
                    msg += (f" — ignorés : {n_skip} lignes, "
                            f"{n_multi} multialléliques")
                st.success(msg)
            except Exception as exc:
                st.error(f"❌ {exc}")


def main():
    init_state()
    st.title("🐄 Bovine SNP Platform")
    st.caption("QC · Structure · Admixture · ROH · FST · Export PLINK/VCF "
               f"— v{APP_VERSION}")

    with st.sidebar:
        _sidebar_data_loader()

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("Missingness SNP (geno)", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["geno"], 0.01)
        mind = st.slider("Missingness individu (mind)", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["mind"], 0.01)
        maf_thr = st.slider("MAF minimale", 0.0, 0.5,
                            DEFAULT_THRESHOLDS["maf"], 0.01)
        hwe_thr = st.number_input("HWE p-value (exclure <)",
                                  value=DEFAULT_THRESHOLDS["hwe"],
                                  format="%.0e")
        het_sd = st.slider("Écarts-types hétérozygotie", 1.0, 5.0,
                           DEFAULT_THRESHOLDS["het_sd"], 0.1)
        params = {"geno": geno, "mind": mind, "maf": maf_thr,
                  "hwe": hwe_thr, "het_sd": het_sd}

        st.divider()
        if st.button("🚀 Pipeline complet", type="primary",
                     **STRETCH):
            if not has_data():
                st.error("Chargez des données d'abord.")
            else:
                st.session_state.run_requested = True
        st.caption(f"v{APP_VERSION} — parser vectorisé · ROH vectorisé · "
                   "FST · NMF · ARS-UCD1.2")

    if st.session_state.get("run_requested"):
        st.session_state.run_requested = False
        try:
            with st.spinner("Pipeline complet en cours..."):
                run_full_pipeline(params)
            st.success("✅ Pipeline terminé — consultez les onglets.")
        except Exception as exc:
            st.error(f"❌ Erreur pipeline : {exc}")

    if not has_data():
        st.info("👉 Générez un jeu de démo ou importez `.ped`+`.map` "
                "ou un `.vcf` depuis la barre latérale.")
        st.markdown("""
        ### Formats supportés
        | Format | Extension | Notes |
        |---|---|---|
        | PLINK texte | `.ped` + `.map` | standard |
        | PLINK gzippé | `.ped.gz` + `.map.gz` | auto-détecté |
        | VCF 4.2 | `.vcf`, `.vcf.gz` | biallélique |
        | PLINK binaire | `.bed`/`.bim`/`.fam` | non supporté en entrée
          (`plink --recode`) |

        ### Modules
        - **QC** : missingness, MAF, HWE exact, hétérozygotie
        - **Structure** : ACP, MDS (IBS), GRM VanRaden
        - **Admixture** : NMF (K ajustable)
        - **Démographie** : LD decay, ROH + FROH + classes de longueur
        - **Sélection** : FST/SNP, Manhattan, FST pairwise
        - **Export** : PLINK `.bed/.bim/.fam`, VCF, rapport HTML
        """)
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "🧬 Structure", "🎨 Admixture",
                    "📈 Démographie", "🔍 Sélection", "📤 Export",
                    "📄 Rapport"])

    # ---------- Aperçu ----------
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())
        c4.metric("Taux d'appel", f"{1 - np.isnan(gt).mean():.4f}")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Individus")
            st.dataframe(ind_df.head(20), **STRETCH)
        with c2:
            st.subheader("SNPs")
            st.dataframe(snp_df.head(20), **STRETCH)

        st.subheader("Effectifs par population")
        counts = (ind_df["FID"].value_counts().rename_axis("Population")
                  .reset_index(name="N"))
        st.plotly_chart(px.bar(counts, x="Population", y="N", height=350)
                        .update_layout(**_LAYOUT), **STRETCH)

        with st.expander("🔬 Compatibilité ARS-UCD1.2"):
            chroms_seen = sorted(snp_df["CHR"].astype(str).unique(),
                                 key=_chr_sort_key)
            cov = []
            for c in chroms_seen:
                key = str(c).upper().replace("CHR", "")
                if key in ARS_UCD12_LENGTHS:
                    mask = snp_df["CHR"].astype(str) == c
                    max_bp = int(snp_df.loc[mask, "BP"].max())
                    cov.append({"CHR": c, "N_SNP": int(mask.sum()),
                                "max_bp": max_bp,
                                "ARS_len": ARS_UCD12_LENGTHS[key],
                                "OK": max_bp <= ARS_UCD12_LENGTHS[key]})
            st.write(f"Chromosomes détectés : {len(chroms_seen)} — "
                     f"reconnus ARS-UCD1.2 : {len(cov)}")
            if cov:
                st.dataframe(pd.DataFrame(cov), **STRETCH)

    # ---------- QC ----------
    with tabs[1]:
        st.subheader("Contrôle qualité")
        if st.button("▶ Lancer le QC", **STRETCH):
            try:
                with st.spinner("Filtrage QC..."):
                    gt_f, ind_f, snp_f, qc_stats = apply_qc_filters(
                        gt, ind_df, snp_df, params)
                st.session_state.gt_filt = gt_f
                st.session_state.ind_filt = ind_f
                st.session_state.snp_filt = snp_f
                st.session_state.qc_stats = qc_stats
                invalidate_downstream()
                st.success("✅ QC terminé.")
            except Exception as exc:
                st.error(f"❌ {exc}")

        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus finaux", s["n_ind_final"],
                      delta=-s["excluded_ind"], delta_color="inverse")
            c2.metric("SNPs finaux", s["n_snp_final"],
                      delta=-s["excluded_snp"], delta_color="inverse")
            c3.metric("Ho / He",
                      f"{s['het_obs']:.3f} / {s['het_exp']:.3f}")
            c4.metric("MAF moyenne", f"{s['maf_mean']:.3f}")

            st.dataframe(pd.DataFrame(s["log"],
                                      columns=["Filtre", "Exclus", "Unité"]),
                         **STRETCH)

            st.markdown("#### Avant filtrage")
            st.plotly_chart(plot_missingness_dashboard(
                missingness_per_ind(gt), missingness_per_snp(gt)),
                **STRETCH)
            c1, c2 = st.columns(2)
            c1.plotly_chart(plot_hist(maf(gt), "Spectre MAF (brut)", "MAF",
                                      "#2ecc71"), **STRETCH)
            c2.plotly_chart(plot_hist(heterozygosity(gt),
                                      "Hétérozygotie observée (brute)",
                                      "Ho", "#9b59b6"),
                            **STRETCH)

            st.markdown("#### Après filtrage")
            gtf = st.session_state.gt_filt
            c1, c2 = st.columns(2)
            c1.plotly_chart(plot_hist(maf(gtf), "Spectre MAF (post-QC)",
                                      "MAF", "#27ae60"),
                            **STRETCH)
            c2.plotly_chart(plot_hist(inbreeding_f_hat(gtf),
                                      "F = 1 − Ho/He (post-QC)", "F",
                                      "#e67e22"), **STRETCH)
        else:
            st.info("Cliquez sur **Lancer le QC**.")

    # ---------- Structure ----------
    with tabs[2]:
        st.subheader("Structure des populations")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if st.button("▶ ACP + MDS + GRM", **STRETCH):
                try:
                    gtf = st.session_state.gt_filt
                    with st.spinner("ACP..."):
                        s_, v_ = pca_analysis(gtf, 10)
                        st.session_state.pca_scores = s_
                        st.session_state.pca_var = v_
                    with st.spinner("MDS (IBS)..."):
                        st.session_state.mds_coords = mds_analysis(gtf, 5)
                    with st.spinner("GRM..."):
                        st.session_state.kinship = kinship_matrix(gtf)
                    st.success("✅ Terminé.")
                except Exception as exc:
                    st.error(f"❌ {exc}")

            labels = as_str_array(st.session_state.ind_filt["FID"])
            if st.session_state.pca_scores is not None:
                scores = st.session_state.pca_scores
                var = st.session_state.pca_var
                n_pc = scores.shape[1]
                c1, c2 = st.columns(2)
                pc_x = c1.selectbox("Axe X", range(n_pc), 0,
                                    format_func=lambda i: f"PC{i+1}")
                pc_y = c2.selectbox("Axe Y", range(n_pc),
                                    1 if n_pc > 1 else 0,
                                    format_func=lambda i: f"PC{i+1}")
                st.plotly_chart(plot_pca(scores, var, labels, pc_x, pc_y),
                                **STRETCH)
                st.plotly_chart(plot_scree(var), **STRETCH)
            if st.session_state.mds_coords is not None:
                st.plotly_chart(plot_mds(st.session_state.mds_coords, labels),
                                **STRETCH)
            if st.session_state.kinship is not None:
                ids = as_str_array(
                    st.session_state.ind_filt["FID"].astype(str) + "_"
                    + st.session_state.ind_filt["IID"].astype(str))
                st.plotly_chart(plot_kinship_heatmap(
                    st.session_state.kinship, ids), **STRETCH)
                G = st.session_state.kinship
                iu = np.triu_indices(G.shape[0], k=1)
                pairs = pd.DataFrame({
                    "Ind_1": ids[iu[0]], "Ind_2": ids[iu[1]],
                    "Parenté": G[iu]}).nlargest(15, "Parenté")
                st.subheader("Paires les plus apparentées")
                st.dataframe(pairs.round(4), **STRETCH)

    # ---------- Admixture ----------
    with tabs[3]:
        st.subheader("Proportions d'ancestralité (NMF)")
        st.caption("Approximation non supervisée par factorisation "
                   "non-négative — à interpréter avec prudence, ce n'est "
                   "pas le modèle de vraisemblance d'ADMIXTURE.")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            c1, c2 = st.columns([1, 2])
            K = c1.slider("Nombre d'ancestralités (K)", 2, 10, 4)
            max_iter = c2.slider("Itérations NMF", 100, 2000, 500, 100)
            if st.button("▶ Calculer l'admixture", **STRETCH):
                try:
                    with st.spinner(f"NMF K={K}..."):
                        Q, _ = admixture_nmf(st.session_state.gt_filt,
                                             K=int(K), max_iter=int(max_iter))
                    st.session_state.admixture_Q = Q
                    st.session_state.admixture_K = int(K)
                    st.success(f"✅ Matrice Q : {Q.shape}")
                except Exception as exc:
                    st.error(f"❌ {exc}")

            if st.session_state.admixture_Q is not None:
                Q = st.session_state.admixture_Q
                Kc = st.session_state.admixture_K
                pops = as_str_array(st.session_state.ind_filt["FID"])
                iids = as_str_array(st.session_state.ind_filt["IID"])
                fig, df_sorted = plot_admixture(Q, iids, pops, Kc)
                st.plotly_chart(fig, **STRETCH)
                st.subheader("Proportions moyennes par population")
                st.dataframe(
                    df_sorted.groupby("Pop")[[f"K{k+1}" for k in range(Kc)]]
                    .mean().round(3), **STRETCH)

    # ---------- Démographie ----------
    with tabs[4]:
        st.subheader("Démographie")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            sub1, sub2 = st.tabs(["LD decay", "ROH / consanguinité"])
            snp_f = st.session_state.snp_filt
            gtf = st.session_state.gt_filt

            with sub1:
                c1, c2 = st.columns(2)
                max_kb = c1.slider("Distance max (kb)", 100, 5000, 1000, 100)
                max_snp = c2.slider("SNPs échantillonnés", 200, 5000, 1500, 100)
                if st.button("▶ Calculer le LD", key="btn_ld",
                             **STRETCH):
                    try:
                        with st.spinner("LD decay..."):
                            st.session_state.ld_df = ld_decay(
                                gtf, as_str_array(snp_f["CHR"]),
                                as_int_array(snp_f["BP"]),
                                max_kb=float(max_kb), max_snp=int(max_snp))
                        st.success(f"✅ {len(st.session_state.ld_df):,} paires")
                    except Exception as exc:
                        st.error(f"❌ {exc}")
                ld_df = st.session_state.ld_df
                if ld_df is not None and not ld_df.empty:
                    fig = plot_ld_decay(ld_df)
                    if fig:
                        st.plotly_chart(fig, **STRETCH)
                    c1, c2 = st.columns(2)
                    near = ld_df[ld_df["dist_kb"] <= 100]["r2"]
                    far = ld_df[ld_df["dist_kb"] > 500]["r2"]
                    c1.metric("r² moyen < 100 kb",
                              f"{near.mean():.3f}" if len(near) else "n/a")
                    c2.metric("r² moyen > 500 kb",
                              f"{far.mean():.3f}" if len(far) else "n/a")

            with sub2:
                c1, c2 = st.columns(2)
                min_snps = c1.slider("SNPs minimum par ROH", 5, 200, 30, 5)
                min_kb = c2.slider("Longueur minimale (kb)", 50, 5000, 500, 50)
                if st.button("▶ Détecter les ROH", key="btn_roh",
                             **STRETCH):
                    try:
                        with st.spinner("Détection des ROH..."):
                            roh_df, froh = detect_roh(
                                gtf, as_str_array(snp_f["CHR"]),
                                as_int_array(snp_f["BP"]),
                                min_snps=int(min_snps), min_kb=float(min_kb))
                        st.session_state.roh_df = roh_df
                        st.session_state.froh = froh
                        st.success(f"✅ {len(roh_df)} segments ROH détectés")
                    except Exception as exc:
                        st.error(f"❌ {exc}")

                if st.session_state.froh is not None:
                    froh = st.session_state.froh
                    roh_df = st.session_state.roh_df
                    labels = as_str_array(st.session_state.ind_filt["FID"])
                    iids = as_str_array(st.session_state.ind_filt["IID"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("FROH moyen", f"{np.mean(froh):.4f}")
                    c2.metric("FROH médian", f"{np.median(froh):.4f}")
                    c3.metric("Segments ROH", len(roh_df))
                    st.plotly_chart(plot_roh_histogram(froh, labels),
                                    **STRETCH)
                    if len(roh_df) > 0:
                        fig = plot_roh_length_classes(roh_df)
                        if fig:
                            st.plotly_chart(fig, **STRETCH)
                        top_idx = np.argsort(-froh)[:10]
                        st.subheader("Individus les plus consanguins")
                        st.dataframe(pd.DataFrame({
                            "IID": iids[top_idx], "Pop": labels[top_idx],
                            "FROH": np.round(froh[top_idx], 4)}),
                            **STRETCH)
                        st.subheader("FROH moyen par population")
                        st.dataframe(pd.DataFrame({"Pop": labels,
                                                   "FROH": froh})
                                     .groupby("Pop")["FROH"]
                                     .agg(["mean", "median", "max"])
                                     .round(4), **STRETCH)
                        st.plotly_chart(plot_roh_manhattan(roh_df, iids),
                                        **STRETCH)

    # ---------- Sélection ----------
    with tabs[5]:
        st.subheader("Signatures de sélection")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gtf = st.session_state.gt_filt
            snp_f = st.session_state.snp_filt
            labels = as_str_array(st.session_state.ind_filt["FID"])
            sub1, sub2 = st.tabs(["FST par SNP", "FST pairwise"])

            with sub1:
                threshold_q = st.slider("Quantile des outliers", 0.95, 0.9999,
                                        0.999, 0.0001, format="%.4f")
                if st.button("▶ Calculer le FST par SNP", key="btn_fst",
                             **STRETCH):
                    try:
                        with st.spinner("FST..."):
                            st.session_state.fst = fst_per_snp(gtf, labels)
                        st.success("✅ FST calculé.")
                    except Exception as exc:
                        st.error(f"❌ {exc}")

                if st.session_state.fst is not None:
                    fst = st.session_state.fst
                    clean = fst[np.isfinite(fst)]
                    if len(clean):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("FST moyen", f"{clean.mean():.4f}")
                        c2.metric("FST médian", f"{np.median(clean):.4f}")
                        thr = float(np.quantile(clean, threshold_q))
                        c3.metric("Seuil outliers", f"{thr:.4f}")
                        fig = plot_manhattan(fst, as_str_array(snp_f["CHR"]),
                                             as_int_array(snp_f["BP"]),
                                             threshold_q=threshold_q)
                        if fig:
                            st.plotly_chart(fig, **STRETCH)
                        out = snp_f.copy()
                        out["FST"] = fst
                        out = (out[out["FST"] >= thr]
                               .sort_values("FST", ascending=False)
                               .head(50))
                        st.subheader("SNPs outliers (top 50)")
                        st.dataframe(
                            out[["CHR", "SNP", "BP", "FST"]].round(4),
                            **STRETCH)
                        st.download_button(
                            "⬇ Exporter les outliers (CSV)",
                            out.to_csv(index=False).encode("utf-8"),
                            file_name="fst_outliers.csv", mime="text/csv")

            with sub2:
                if st.button("▶ Calculer le FST pairwise", key="btn_fstp",
                             **STRETCH):
                    try:
                        with st.spinner("FST pairwise..."):
                            mat, pops = fst_pairwise(gtf, labels)
                        st.session_state.fst_pairwise_matrix = mat
                        st.session_state.fst_pops = pops
                        st.success("✅ Matrice calculée.")
                    except Exception as exc:
                        st.error(f"❌ {exc}")
                if st.session_state.fst_pairwise_matrix is not None:
                    st.plotly_chart(
                        plot_fst_pairwise(st.session_state.fst_pairwise_matrix,
                                          st.session_state.fst_pops),
                        **STRETCH)

    # ---------- Export ----------
    with tabs[6]:
        st.subheader("Export des données post-QC")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            prefix = st.text_input("Préfixe des fichiers", "bovine_qc")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("📦 Préparer PLINK", **STRETCH):
                    try:
                        with st.spinner("Génération PLINK..."):
                            st.session_state["_plink_zip"] = build_plink_zip(
                                st.session_state.gt_filt,
                                st.session_state.ind_filt,
                                st.session_state.snp_filt, prefix=prefix)
                        st.success("✅ Prêt.")
                    except Exception as exc:
                        st.error(f"❌ {exc}")
                if st.session_state.get("_plink_zip"):
                    st.download_button(
                        "⬇ .bed/.bim/.fam (zip)",
                        data=st.session_state["_plink_zip"],
                        file_name=f"{prefix}_plink.zip",
                        mime="application/zip", **STRETCH)
            with c2:
                if st.button("📄 Préparer VCF", **STRETCH):
                    try:
                        with st.spinner("Génération VCF..."):
                            st.session_state["_vcf_str"] = build_vcf_output(
                                st.session_state.gt_filt,
                                st.session_state.ind_filt,
                                st.session_state.snp_filt)
                        st.success("✅ Prêt.")
                    except Exception as exc:
                        st.error(f"❌ {exc}")
                if st.session_state.get("_vcf_str"):
                    st.download_button(
                        "⬇ .vcf",
                        data=st.session_state["_vcf_str"].encode("utf-8"),
                        file_name=f"{prefix}.vcf", mime="text/plain",
                        **STRETCH)
            with c3:
                st.metric("Individus exportés",
                          st.session_state.gt_filt.shape[0])
                st.metric("SNPs exportés",
                          st.session_state.gt_filt.shape[1])
                st.download_button(
                    "⬇ Matrice de dosages (CSV)",
                    pd.DataFrame(
                        st.session_state.gt_filt,
                        index=st.session_state.ind_filt["IID"],
                        columns=st.session_state.snp_filt["SNP"]
                    ).to_csv().encode("utf-8"),
                    file_name=f"{prefix}_dosages.csv", mime="text/csv",
                    **STRETCH)

    # ---------- Rapport ----------
    with tabs[7]:
        st.subheader("Rapport HTML")
        if not has_qc():
            st.warning("⚠️ Lancez au moins le QC.")
        else:
            project_name = st.text_input("Nom du projet", "Cattle_Project")
            if st.button("📄 Générer le rapport", **STRETCH):
                with st.spinner("Assemblage du rapport..."):
                    figures, tables = collect_figures()
                    html = build_report_html(
                        {"project_name": project_name},
                        st.session_state.qc_stats, figures, tables)
                st.download_button(
                    "⬇ Télécharger le rapport",
                    data=html.encode("utf-8"),
                    file_name=(f"rapport_bovine_"
                               f"{datetime.now():%Y%m%d_%H%M}.html"),
                    mime="text/html", **STRETCH)
                st.success("✅ Rapport prêt.")
                with st.expander("Prévisualisation"):
                    render_html(html, height=700)


if __name__ == "__main__":
    main()
