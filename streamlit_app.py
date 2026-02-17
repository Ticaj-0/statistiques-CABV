import os
import re
import streamlit as st
import pandas as pd
import html as html_lib
import streamlit.components.v1 as components

ANNEE_SAISON = 2025

# ============================================================
# Logo grand et centré pendant le chargement
# ============================================================
logo_placeholder = st.empty()

with logo_placeholder.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo_CABV2.png", width=500)

# ============================================================
# 0) Fichiers / cache (Parquet direct)
#    -> puisque tu uploades déjà les .parquet sur Streamlit Cloud
# ============================================================
def file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def read_parquet_cached(path: str, mtime: float) -> pd.DataFrame:
    """
    Lecture parquet + cache Streamlit.
    'mtime' force l’invalidation du cache si tu remplaces le fichier sur le serveur.
    """
    return pd.read_parquet(path)


# ============================================================
# 1) Normalisations / parsing
# ============================================================
def convertir_temps_en_secondes(valeur):
    if isinstance(valeur, str) and re.match(r"^\d+:\d+\.\d+$", valeur):
        minutes, reste = valeur.split(":")
        return int(minutes) * 60 + float(reste)
    try:
        return float(valeur)
    except Exception:
        return None


def norm_txt(s):
    return str(s).strip().lower()


def kind_records(perf_str):
    s = str(perf_str).strip().lower().replace(" ", "")
    if s.endswith("pts"):
        return "p"
    if "s" in s or ":" in s or "'" in s or "″" in s or "′" in s or "’" in s:
        return "t"
    if re.match(r"^\d+(\.\d+)?m\d+$", s):  # 2m40
        return "d"
    return "t"


def est_discipline_mesure(discipline_raw: str) -> bool:
    if pd.isna(discipline_raw):
        return False
    s = str(discipline_raw).strip().lower()
    mots_mesure = ["hauteur", "longueur", "perche", "triple", "poids", "disque", "javelot", "marteau", "balle"]
    return any(m in s for m in mots_mesure)


def est_discipline_points(discipline_raw: str) -> bool:
    if pd.isna(discipline_raw):
        return False
    s = str(discipline_raw).strip().lower()
    return any(x in s for x in ["athlon", "tétrathlon", "pentathlon", "heptathlon", "décathlon", "decathlon"])


def performance_to_key(val, kind: str):
    """
    Retourne une clé comparable:
      - temps -> ("t", centièmes)
      - distance -> ("d", centimètres)
      - points -> ("p", points entiers)
    """
    if pd.isna(val):
        return None

    s = str(val).strip().lower()
    s = s.replace(" ", "").replace("*", "")
    s = s.replace(",", ".")
    s = s.replace("’", "'").replace("′", "'").replace("″", '"')

    # points
    if s.endswith("pts"):
        try:
            return ("p", int(float(s[:-3])))
        except Exception:
            return None

    # formats type 3'13"00
    s = s.replace("'", "m").replace('"', "s")

    # format minute.seconde.centiemes : 1.34.20 -> 1:34.20
    if re.match(r"^\d+\.\d{2}\.\d{2}$", s):
        s = s.replace(".", ":", 1)

    # mm:ss.xx
    if ":" in s:
        try:
            mn, sec = s.split(":")
            total = int(mn) * 60 + float(sec)
            return ("t", int(round(total * 100)))
        except Exception:
            return None

    # 1m11s11 / 10s62 / 8m49s21 / 1h02m03s45
    if "s" in s:
        m = re.match(r"^(?:(\d+)h)?(?:(\d+)m)?(\d+)s(\d+)?$", s)
        if m:
            h = int(m.group(1) or 0)
            mn = int(m.group(2) or 0)
            sec = int(m.group(3) or 0)
            frac = m.group(4)
            frac_val = (int(frac) / (10 ** len(frac))) if frac else 0.0
            total = h * 3600 + mn * 60 + sec + frac_val
            return ("t", int(round(total * 100)))

    # distance type 2m40 / 46m74 / 40m00
    if re.match(r"^\d+(\.\d+)?m\d+$", s):
        try:
            metres = float(s.replace("m", ".", 1))
            return ("d", int(round(metres * 100)))
        except Exception:
            return None

    # nombre simple
    try:
        x = float(s)
        if kind == "t":
            return ("t", int(round(x * 100)))
        if kind == "d":
            return ("d", int(round(x * 100)))
        if kind == "p":
            return ("p", int(round(x)))
        return None
    except Exception:
        return None


def normaliser_categorie_records(groupe):
    s = str(groupe).strip().lower()
    s = s.replace("femmes", "femme").replace("hommes", "homme")
    s = re.sub(r"\bu(\d+)\s*w\b", r"u\1 femme", s)
    s = re.sub(r"\bu(\d+)\s*m\b", r"u\1 homme", s)
    if s in ["homme", "femme"]:
        s = "adulte " + s
    return s


def normaliser_discipline(d):
    if pd.isna(d):
        return ""

    s = str(d).strip().lower()
    s = s.replace("\u00a0", " ").replace("×", "x")
    s = s.replace(",", ".")
    s = s.replace("’", "'")

    # enlever parenthèses/crochets
    s = re.sub(r"[\(\)\[\]]", " ", s)

    # supprimer séparateurs de milliers entre chiffres: 1'000 -> 1000
    s = re.sub(r"(?<=\d)'\s*(?=\d)", "", s)

    # normaliser relais
    s = re.sub(r"\s*x\s*", "x", s)

    # haies: supprimer hauteur finale
    s = re.sub(r"(haies)\s*\d+(\.\d+)?\s*$", r"\1", s)

    # "gr." / "gr" -> "g"
    s = re.sub(r"\bgr\.?\b", "g", s)
    s = s.replace("g.", "g")

    # supprimer espaces
    s = re.sub(r"\s+", "", s)

    return s


def calcul_categorie(age):
    if age >= 20:
        return "Adulte"
    elif age >= 18:
        return "U20"
    elif age >= 16:
        return "U18"
    elif age >= 14:
        return "U16"
    elif age >= 12:
        return "U14"
    elif age >= 10:
        return "U12"
    elif age >= 8:
        return "U10"
    else:
        return "Moins de 8 ans"


def compter_decimales_temps(s: str) -> int:
    if s is None:
        return 0
    s = str(s).strip()
    if s.lower().endswith("m"):
        s = s[:-1].strip()
    s = s.replace("'", ":").replace('"', ".").replace(",", ".")
    sec_part = s.split(":")[-1]
    return len(sec_part.split(".")[1]) if "." in sec_part else 0


def ajuster_temps_manuel_pour_classement(perf_seconds: float, nb_decimales: int, manuel: bool) -> float:
    if perf_seconds is None or pd.isna(perf_seconds):
        return perf_seconds
    if not manuel:
        return perf_seconds
    if nb_decimales == 0:
        return perf_seconds + 0.99
    if nb_decimales == 1:
        return perf_seconds + 0.09
    return perf_seconds


def texte_complement_discipline(discipline, categorie_selection, type_salle):
    if str(type_salle).lower() == "indoor":
        return " (depuis 2006)"
    if not str(categorie_selection).startswith("U10"):
        return ""
    if discipline == "50m":
        return " (depuis 2016)"
    if discipline == "600m":
        return " (depuis 2017)"
    return ""


# ============================================================
# 2) Chargements & pré-calculs (optimisés)
# ============================================================
@st.cache_data(show_spinner=False)
def charger_et_preparer_donnees(path_parquet: str, mtime: float) -> pd.DataFrame:
    df = read_parquet_cached(path_parquet, mtime).copy()

    # IMPORTANT: renommer colonnes (si ton parquet garde les noms Excel originaux, adapte ici)
    df.columns = [
        "discipline", "performance", "nom", "nationalite", "naissance", "lieu",
        "date", "categorie_fichier", "sexe", "record", "multiple", "salle"
    ]

    df["sexe"] = df["sexe"].replace({"H": "Homme", "F": "Femme"})
    df["performance_brute"] = df["performance"]

    df = df.dropna(subset=["discipline", "performance", "nom"]).copy()

    df["performance_sans_etoile"] = df["performance_brute"].astype(str).str.replace("*", "", regex=False).str.strip()
    df["manuel"] = df["performance_sans_etoile"].astype(str).str.lower().str.endswith("m")
    df["performance_sans_m"] = df["performance_sans_etoile"].astype(str).str[:-1].where(
        df["manuel"], df["performance_sans_etoile"]
    ).astype(str).str.strip()

    df["performance"] = df["performance_sans_m"].apply(convertir_temps_en_secondes)
    df["performance"] = pd.to_numeric(df["performance"], errors="coerce")

    df["decimales_temps"] = df["performance_sans_etoile"].apply(compter_decimales_temps)
    df["performance_classement"] = [
        ajuster_temps_manuel_pour_classement(p, d, m)
        for p, d, m in zip(df["performance"], df["decimales_temps"], df["manuel"])
    ]

    def extraire_annee(val):
        if pd.isnull(val):
            return None
        if isinstance(val, (int, float)) and 1900 <= int(val) <= 2100:
            return int(val)
        val_str = str(val).strip()
        if re.match(r"^\d{4}$", val_str):
            return int(val_str)
        try:
            return pd.to_datetime(val_str, errors="coerce").year
        except Exception:
            return None

    df["annee_competition"] = df["date"].apply(extraire_annee)
    df["annee_naissance"] = df["naissance"].apply(extraire_annee)
    df["age"] = df["annee_competition"] - df["annee_naissance"]

    df["categorie_calculee"] = df["age"].apply(lambda x: calcul_categorie(x) if pd.notnull(x) else None)
    df["categorie"] = df["categorie_calculee"]
    df.loc[df["categorie"].isnull(), "categorie"] = df.loc[df["categorie"].isnull(), "categorie_fichier"]
    df["categorie"] = df["categorie"].fillna("Adulte")

    df["categorie_sexe"] = df["categorie"] + " " + df["sexe"]
    df["salle"] = df["salle"].fillna("")

    # PRÉ-CALCULS (énorme gain)
    df["discipline_norm"] = df["discipline"].apply(normaliser_discipline)
    df["cat_norm"] = (df["categorie"].astype(str) + " " + df["sexe"].astype(str)).apply(norm_txt)

    unique_disc = df["discipline"].dropna().unique().tolist()
    kind_map = {}
    for d in unique_disc:
        if est_discipline_points(d):
            kind_map[d] = "p"
        elif est_discipline_mesure(d):
            kind_map[d] = "d"
        else:
            kind_map[d] = "t"
    df["kind"] = df["discipline"].map(kind_map).fillna("t")

    df["perf_key"] = [
        performance_to_key(v, k) for v, k in zip(df["performance_sans_m"].tolist(), df["kind"].tolist())
    ]

    return df


@st.cache_data(show_spinner=False)
def charger_records_lookup(path_parquet: str, mtime: float) -> pd.DataFrame:
    df_records = read_parquet_cached(path_parquet, mtime).copy()

    df_records["discipline_norm"] = df_records["Discipline"].apply(normaliser_discipline)
    df_records["cat_norm"] = df_records["Groupe"].apply(normaliser_categorie_records)
    df_records["perf_key"] = df_records["Performance"].apply(lambda v: performance_to_key(v, kind_records(v)))

    df_records = df_records.dropna(subset=["Rang", "discipline_norm", "cat_norm", "perf_key"]).copy()
    df_records["Rang"] = df_records["Rang"].astype(int)
    df_records["Groupe"] = df_records["Groupe"].astype(str).str.strip()

    return df_records[["discipline_norm", "cat_norm", "perf_key", "Rang", "Groupe"]].copy()


def appliquer_records_outdoor_fast(df: pd.DataFrame, df_records_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    salle_norm = df["salle"].astype(str).str.strip().str.lower()
    mask_out = salle_norm.eq("outdoor")

    df["record"] = df["record"].fillna("")
    df.loc[~mask_out, "record"] = ""

    if mask_out.sum() == 0 or df_records_lookup.empty:
        return df

    sub = df.loc[mask_out].copy().reset_index(drop=False).rename(columns={"index": "_idx"})

    c1 = sub[["_idx", "discipline_norm", "cat_norm", "perf_key"]].copy()
    c1["prio"] = 0

    mask_u20f = c1["cat_norm"].eq("u20 femme")
    c2 = c1.loc[mask_u20f].copy()
    c2["cat_norm"] = "adulte femme"
    c2["prio"] = 1

    candidats = pd.concat([c1, c2], ignore_index=True)

    merged = candidats.merge(
        df_records_lookup,
        how="left",
        on=["discipline_norm", "cat_norm", "perf_key"],
    ).dropna(subset=["Rang", "Groupe"])

    if merged.empty:
        df.loc[mask_out, "record"] = ""
        return df

    merged = merged.sort_values(["_idx", "prio"]).drop_duplicates(subset=["_idx"], keep="first")

    rang = merged["Rang"].astype(int)
    groupe = merged["Groupe"].astype(str)
    groupe_norm = groupe.apply(normaliser_categorie_records)

    merged["record_txt"] = rang.astype(str) + "e MP VS"
    mask_add_cat_mp = (~groupe_norm.isin(["adulte homme", "adulte femme"])) & (~rang.eq(1))
    merged.loc[mask_add_cat_mp, "record_txt"] = merged.loc[mask_add_cat_mp, "record_txt"] + " " + groupe[mask_add_cat_mp]

    merged.loc[rang.eq(1), "record_txt"] = "Record VS"
    mask_add_cat_record = rang.eq(1) & (~groupe_norm.isin(["adulte homme", "adulte femme"]))
    merged.loc[mask_add_cat_record, "record_txt"] = "Record VS " + groupe[mask_add_cat_record]

    sub = sub.merge(merged[["_idx", "record_txt"]], on="_idx", how="left")
    sub["record"] = sub["record_txt"].fillna("")

    df.loc[sub["_idx"], "record"] = sub["record"].values
    return df


@st.cache_data(show_spinner=False)
def charger_limites_lookup(path_parquet: str, mtime: float) -> dict:
    df_limites = read_parquet_cached(path_parquet, mtime).copy()
    df_limites["Saison"] = df_limites["Saison"].astype(str).str.strip().str.lower()
    df_limites["Discipline"] = df_limites["Discipline"].astype(str).str.strip()
    df_limites["Categorie"] = df_limites["Categorie"].astype(str).str.strip()
    df_limites["Sexe"] = df_limites["Sexe"].astype(str).str.strip()

    lookup = {}
    for _, r in df_limites.iterrows():
        lookup[(r["Discipline"], r["Categorie"], r["Sexe"], r["Saison"])] = r.get("Limite", None)
    return lookup


@st.cache_data(show_spinner=False)
def charger_doublons_valides_set(path_parquet: str, mtime: float) -> set:
    if not os.path.exists(path_parquet):
        return set()

    df = read_parquet_cached(path_parquet, mtime).copy()
    colonnes = ["performance_affichee", "discipline", "nom", "naissance", "lieu", "date"]
    if df.empty or any(c not in df.columns for c in colonnes):
        return set()

    def normaliser_date_cle(val):
        if pd.isna(val):
            return ""
        s = str(val).strip().replace(".", "/").replace("-", "/")
        if "0000" in s:
            return "0000-01-01"
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return dt.strftime("%Y-%m-%d") if not pd.isna(dt) else s

    for col in colonnes:
        if col in ["date", "naissance"]:
            df[col] = df[col].apply(normaliser_date_cle)
        else:
            df[col] = df[col].astype(str).str.strip()

    return set(map(tuple, df[colonnes].values))


@st.cache_data(show_spinner=False)
def chargement_des_donnees():
    path_club = "Statistiques_2025.parquet"
    path_records = "records_valaisans_2025.parquet"
    path_limites = "limites.parquet"
    path_doublons = "doublons_valides.parquet"

    df = charger_et_preparer_donnees(path_club, file_mtime(path_club))
    df_records_lookup = charger_records_lookup(path_records, file_mtime(path_records))
    df = appliquer_records_outdoor_fast(df, df_records_lookup)

    limites_lookup = charger_limites_lookup(path_limites, file_mtime(path_limites))
    doublons_valides_set = charger_doublons_valides_set(path_doublons, file_mtime(path_doublons))

    return df, limites_lookup, doublons_valides_set


# ============================================================
# 3) Discipline list (cache)
# ============================================================
ORDRE_DISCIPLINE = [
    "50m","60m","80m","100m","200m","300m","400m","600m","800m","1000m","1500m","2000m","3000m","5000m","10000m",
    "50m haies 76.2","50m haies 106.7","60m haies 60.0","60m haies 68.0","60m haies 76.2","60m haies 106.7",
    "80m haies 76.2","100m haies 76.2","100m haies 84.0","110m haies 91.4","110m haies 99.1","110m haies 106.7",
    "300m haies 76.2","300m haies 84.0","300m haies 91.4","400m haies 76.2","400m haies 84.0","400m haies 91.4",
    "2000m steeple","3000m steeple","Hauteur","Longueur","Longueur [zone]","Perche","Triple",
    "Poids 2.5kg","Poids 3kg","Poids 4kg","Poids 5kg","Poids 6kg","Poids 7.26kg",
    "Disque 0.75kg","Disque 1kg","Disque 1.5kg","Disque 1.75kg","Disque 2kg",
    "Javelot 400g","Javelot 500g","Javelot 600g","Javelot 700g","Javelot 800g",
    "Marteau 3kg","Marteau 4kg","Marteau 5kg","Marteau 6kg","Marteau 7.26kg",
    "Balle 200g","Tétrathlon",
    "Pentathlon 1 U14","Pentathlon 2 U14","Décathlon U18","Décathlon U20","Décathlon",
    "Decathlon indoor","Heptathlon U18","Heptathlon U18 (depuis 2014)","Heptathlon U20","Heptathlon","Heptathlon indoor",
    "Ubs Kids Cup","Hexathlon U16",
    "Pentathlon U16","Pentathlon U18","Pentathlon U16 (depuis 2024)",
    "Relai 5x libre","Relai 6x libre","5x80m","4x100m"
]

@st.cache_data(show_spinner=False)
def get_discipline_list(df: pd.DataFrame, type_salle: str, categories_autorisees_full: list) -> list:
    df_temp = df[df["salle"].astype(str).str.lower() == type_salle.lower()]
    poss = df_temp[df_temp["categorie_sexe"].isin(categories_autorisees_full)]["discipline"].dropna().unique().tolist()
    return sorted(
        poss,
        key=lambda x: (ORDRE_DISCIPLINE.index(x) if x in ORDRE_DISCIPLINE else len(ORDRE_DISCIPLINE), x),
    )


# ============================================================
# 4) Chargement UI (spinner + logo)
# ============================================================
with st.spinner("Chargement des statistiques…"):
    df, limites_lookup, doublons_valides_set = chargement_des_donnees()

# Supprimer le logo de chargement
logo_placeholder.empty()

# ============================================================
# Logo petit en haut à gauche après chargement
# ============================================================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("logo_CABV2.png", width=170)

# ============================================================
# 5) Interface
# ============================================================
st.title(f"Statistiques CABV Martigny {ANNEE_SAISON}")
st.subheader("Filtres")

type_salle = st.selectbox("Saison", ["Outdoor", "Indoor"])
categorie_base_order = ["Homme", "Femme", "U20", "U18", "U16", "U14", "U12", "U10"]
sexe_order = ["Homme", "Femme"]
categorie_order = ["Homme", "Femme"] + [
    f"{cat} {sexe}" for cat in categorie_base_order if cat not in ["Homme", "Femme"] for sexe in sexe_order
]
categorie_selection = st.selectbox("Choisir une catégorie", categorie_order)

categorie_base = categorie_selection.split()[0] if " " in categorie_selection else categorie_selection
sexe_sel = categorie_selection.split()[-1]
categorie_hierarchie = ["U10", "U12", "U14", "U16", "U18", "U20", "Adulte", "Homme", "Femme"]
indice = categorie_hierarchie.index(categorie_base)
categories_autorisees = categorie_hierarchie[: indice + 1]
categories_autorisees_full = [f"{cat} {sexe_sel}" for cat in categories_autorisees if cat not in ["Homme", "Femme"]]
if categorie_base in ["Homme", "Femme"]:
    categories_autorisees_full.append(sexe_sel)

discipline_list = get_discipline_list(df, type_salle, categories_autorisees_full)
discipline = st.selectbox("Choisir une discipline",discipline_list)
mode = st.selectbox("Afficher", ["Un seul résultat par athlète", "Tous les résultats"])

# ============================================================
# 6) Recherche & affichage (tout comme avant + optimisé)
# ============================================================
if st.button("Rechercher"):
    df_filtre = df[
        (df["salle"].astype(str).str.lower() == type_salle.lower())
        & (df["discipline"] == discipline)
        & (df["categorie_sexe"].isin(categories_autorisees_full))
    ].copy()

    disciplines_m_mesure = [
        "Hauteur","Longueur [zone]","Longueur","Perche","Triple",
        "Poids 2.5kg","Poids 3kg","Poids 4kg","Poids 5kg","Poids 6kg","Poids 7.26kg",
        "Disque 0.75kg","Disque 1kg","Disque 1.5kg","Disque 1.75kg","Disque 2kg",
        "Javelot 400g","Javelot 500g","Javelot 600g","Javelot 700g","Javelot 800g",
        "Marteau 3kg","Marteau 4kg","Marteau 5kg","Marteau 6kg","Marteau 7.26kg",
        "Balle 200g",
        "Tétrathlon","Pentathlon 1 U14","Pentathlon 2 U14","Décathlon U18","Décathlon U20","Décathlon",
        "Heptathlon U18","Heptathlon U18 (depuis 2014)","Heptathlon U20","Heptathlon",
        "Ubs Kids Cup","Hexathlon U16","Pentathlon U16","Pentathlon U18","Pentathlon U16 (depuis 2024)"
    ]
    disciplines_multiples = [
        "Tétrathlon","Pentathlon 1 U14","Pentathlon 2 U14","Décathlon U18","Décathlon U20","Décathlon",
        "Heptathlon U18","Heptathlon U18 (depuis 2014)","Heptathlon U20","Heptathlon",
        "Ubs Kids Cup","Hexathlon U16","Pentathlon U16","Pentathlon U18","Pentathlon U16 (depuis 2024)",
        "Heptathlon indoor","Pentathlon indoor"
    ]
    relais = ["Relai 5x libre","Relai 6x libre","5x80m","4x100m"]

    # ==========================
    # Un seul résultat par athlète
    # ==========================
    if mode == "Un seul résultat par athlète":
        if discipline in disciplines_m_mesure:
            df_filtre = df_filtre.sort_values("performance", ascending=False).drop_duplicates("nom", keep="first")
        else:
            df_filtre = df_filtre.sort_values("performance_classement", ascending=True).drop_duplicates("nom", keep="first")

    # ==========================
    # Tri initial
    # ==========================
    if discipline in disciplines_m_mesure:
        df_filtre = df_filtre.sort_values("performance", ascending=False).reset_index(drop=True)
    else:
        df_filtre = df_filtre.sort_values("performance_classement", ascending=True).reset_index(drop=True)

    # ==========================
    # Catégorie affichée (comme avant)
    # ==========================
    indice_choisi = categorie_hierarchie.index(categorie_base)

    def afficher_categorie(row):
        if pd.notnull(row.get("categorie_calculee")):
            cat = row["categorie_calculee"]
        elif pd.notnull(row.get("categorie")):
            cat = row["categorie"]
        else:
            cat = "Adulte"

        cat = str(cat).strip().capitalize()

        # Si adulte → on n'affiche rien
        if cat == "Adulte":
            return ""

        # Sinon on affiche seulement si la catégorie est dans la hiérarchie visible
        if cat in categorie_hierarchie:
            if sexe_sel in ["Homme", "Femme"]:
                if categorie_hierarchie.index(cat) <= categorie_hierarchie.index(sexe_sel):
                    return cat
            else:
                if categorie_hierarchie.index(cat) <= indice_choisi:
                    return cat

        return ""

    df_filtre["categorie_affichee"] = df_filtre.apply(afficher_categorie, axis=1)

    # ==========================
    # Format date
    # ==========================
    def format_date_flexible(val):
        if pd.isnull(val):
            return ""
        val_str = str(val).strip()
        if re.match(r"^\d{4}$", val_str):
            return val_str
        dt = pd.to_datetime(val, errors="coerce")
        return dt.strftime("%d/%m/%Y") if not pd.isnull(dt) else val_str

    df_filtre["date"] = df_filtre["date"].apply(format_date_flexible)
    df_filtre["naissance"] = df_filtre["naissance"].apply(format_date_flexible)

    # ==========================
    # Format performance affichée
    # ==========================
    def format_performance(val, discipline_):
        val_str = str(val)
        suffix = "*" if "*" in val_str else ""
        val_str = val_str.replace("*", "")

        if discipline_ in disciplines_multiples:
            try:
                return f"{int(float(val_str))}" + suffix
            except Exception:
                return val_str + suffix

        if re.match(r"^\d+:\d+\.\d+$", val_str):
            return val_str + suffix

        try:
            return f"{float(val_str):.2f}" + suffix
        except Exception:
            return val_str + suffix

    df_filtre["performance_affichee"] = df_filtre["performance_brute"].apply(lambda x: format_performance(x, discipline))

    # ==========================
    # Dé-doublonnage (comme avant) + lookup set (plus rapide)
    # ==========================
    colonnes_dedoublonnage = ["performance_affichee", "discipline", "nom", "naissance", "lieu", "date"]

    def normaliser_date_cle(val):
        if pd.isna(val):
            return ""
        s = str(val).strip().replace(".", "/").replace("-", "/")
        if "0000" in s:
            return "0000-01-01"
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return dt.strftime("%Y-%m-%d") if not pd.isna(dt) else s

    def normaliser_colonnes_cle(dfx):
        dfx = dfx.copy()
        for col in colonnes_dedoublonnage:
            if col in ["date", "naissance"]:
                dfx[col] = dfx[col].apply(normaliser_date_cle)
            else:
                dfx[col] = dfx[col].astype(str).str.strip()
        return dfx

    df_filtre_norm = normaliser_colonnes_cle(df_filtre)
    df_filtre_norm["_cle"] = list(map(tuple, df_filtre_norm[colonnes_dedoublonnage].values))
    df_filtre_norm["est_doublon"] = df_filtre_norm.duplicated(subset=colonnes_dedoublonnage, keep=False)
    df_filtre_norm["groupe_valide"] = df_filtre_norm["_cle"].isin(doublons_valides_set)
    df_filtre_norm["_rang"] = df_filtre_norm.groupby("_cle").cumcount()

    a_garder = (df_filtre_norm["groupe_valide"] & (df_filtre_norm["_rang"] < 2)) | (
        ~df_filtre_norm["groupe_valide"] & (df_filtre_norm["_rang"] == 0)
    )
    df_filtre = df_filtre.loc[df_filtre_norm.index[a_garder]].copy().reset_index(drop=True)

    # ==========================
    # Limites (comme avant, mais en lookup dict)
    # ==========================
    HIERARCHIE_LIMITES = ["U10", "U12", "U14", "U16", "U18", "U20", "Adulte"]

    def get_limite(discipline_, categorie_, genre_, saison_):
        discipline_ = str(discipline_).strip()
        categorie_ = str(categorie_).strip()
        genre_ = str(genre_).strip()
        saison_norm = str(saison_).strip().lower()

        cats_a_tester = [categorie_]
        if categorie_ in HIERARCHIE_LIMITES:
            idx = HIERARCHIE_LIMITES.index(categorie_)
            cats_a_tester += HIERARCHIE_LIMITES[idx + 1 :]

        for cat_test in cats_a_tester:
            key = (discipline_, cat_test, genre_, saison_norm)
            if key in limites_lookup:
                return limites_lookup[key], cat_test
        return None, None

    categorie_sel_clean = categorie_selection.split()[0] if " " in categorie_selection else categorie_selection
    limite, categorie_limite_utilisee = get_limite(discipline, categorie_sel_clean, sexe_sel, type_salle)

    def convertir_limite(limite_brute):
        if limite_brute is None or (isinstance(limite_brute, float) and pd.isna(limite_brute)):
            return None

        s = str(limite_brute).strip().lower()
        s = s.replace(",", ".")
        s = s.replace("'", ":").replace('"', ".")

        # nombre simple
        if re.match(r"^\d+(\.\d+)?$", s):
            return float(s)

        # mm:ss.xx
        if re.match(r"^\d+:\d+(\.\d+)?$", s):
            return convertir_temps_en_secondes(s)

        # mètres : 40m00
        if re.match(r"^\d+(\.\d+)?m\d*$", s):
            s2 = s.replace("m", ".")
            try:
                return float(s2)
            except Exception:
                return None

        # points : 2000pts
        if re.match(r"^\d+(\.\d+)?pts$", s):
            try:
                return float(s.replace("pts", ""))
            except Exception:
                return None

        return None

    limite_val = convertir_limite(limite)

    # ==========================
    # Mode mixte (comme avant)
    # ==========================
    if mode == "Tous les résultats" and limite_val is not None and not df_filtre.empty:
        if discipline in disciplines_m_mesure:
            passe = df_filtre["performance"] >= limite_val
            meilleur_asc = False
        else:
            passe = df_filtre["performance_classement"] <= limite_val
            meilleur_asc = True

        df_tmp = df_filtre.copy()
        df_tmp["_passe_limite"] = passe

        noms_qualifies = df_tmp.groupby("nom")["_passe_limite"].any()
        noms_qualifies = noms_qualifies[noms_qualifies].index

        df_qualifies = df_tmp[df_tmp["nom"].isin(noms_qualifies) & df_tmp["_passe_limite"]]
        df_nonqualif = df_tmp[~df_tmp["nom"].isin(noms_qualifies)]
        df_nonqualif_best = (
            df_nonqualif.sort_values("performance_classement", ascending=meilleur_asc)
            .drop_duplicates("nom", keep="first")
        )

        df_filtre = pd.concat([df_qualifies, df_nonqualif_best], ignore_index=True)
        df_filtre = df_filtre.sort_values("performance_classement", ascending=meilleur_asc).reset_index(drop=True)
        df_filtre = df_filtre.drop(columns=["_passe_limite"], errors="ignore")

    # ==========================
    # TITRE (comme avant)
    # ==========================
    texte_plus = texte_complement_discipline(discipline, categorie_selection, type_salle)

    if mode == "Tous les résultats" and limite is not None:
        if discipline in disciplines_m_mesure:
            st.subheader(f"Résultats au dessus de {limite} + Top 200 - {discipline}{texte_plus} - {categorie_selection}")
        else:
            st.subheader(f"Résultats en dessous de {limite} + Top 200 - {discipline}{texte_plus}  - {categorie_selection}")
    else:
        st.subheader(f"{mode} - {discipline}{texte_plus}  - {categorie_selection}")

    # ==========================
    # Classement ex-aequo (comme avant)
    # ==========================
    if discipline in disciplines_m_mesure:
        df_filtre = df_filtre.sort_values("performance", ascending=False).reset_index(drop=True)
        df_filtre["rang"] = df_filtre["performance"].rank(method="min", ascending=False).astype(int)
    else:
        df_filtre = df_filtre.sort_values("performance_classement", ascending=True).reset_index(drop=True)
        df_filtre["rang"] = df_filtre["performance_classement"].rank(method="min", ascending=True).astype(int)

    # ============================================================
    # ✅ Afficher MP / Record UNE SEULE FOIS par athlète
    #    (on garde la meilleure perf, car le tri est déjà fait)
    # ============================================================
    mask_rec = df_filtre["record"].fillna("").astype(str).str.strip().ne("")
    if mask_rec.any():
        dup = df_filtre.loc[mask_rec].duplicated(
            subset=["nom", "record"],
            keep="first"
        )
        idx_to_clear = df_filtre.loc[mask_rec].index[dup]
        df_filtre.loc[idx_to_clear, "record"] = ""

    if df_filtre.empty:
        st.warning("Aucun résultat trouvé pour cette sélection.")

    # ==========================
    # HTML (comme avant)
    # ==========================
    elif discipline in disciplines_multiples or discipline in relais:
        colonnes = ["rang", "nom", "naissance", "performance_affichee", "lieu", "date", "categorie_affichee", "record", "multiple"]
        filtre_affichage = df_filtre[colonnes].rename(columns={
            "nom": "Prénom Nom",
            "naissance": "Naissance",
            "performance_affichee": "Performance",
            "lieu": "Lieu",
            "date": "Date",
            "categorie_affichee": "Cat.",
            "record": "Record",
            "multiple": "Détails"
        })

        details_disciplines_multiples = {
            "Décathlon": "100m, Longueur, Poids (7.26), Hauteur, 400m, 110m haies (106.7), Disque (2), Perche, Javelot (800), 1500m",
            "Décathlon U18":"100m, Longueur, Poids (5), Hauteur, 400m, 110m haies (91.4), Disque (1.5), Perche, Javelot (700), 1500m",
            "Décathlon U20":"100m, Longueur, Poids (6), Hauteur, 400m, 110m haies (99.1), Disque (1.75), Perche, Javelot (800), 1500m",
            "Heptathlon U18":"100m haies (76.2), Hauteur, Poids (3), 200m, Longueur, Javelot (600), 800m",
            "Heptathlon U18 (depuis 2014)":"100m haies (76.2), Hauteur, Poids (3), 200m, Longueur, Javelot (500), 800m",
            "Heptathlon U20":"100m haies (84.0), Hauteur, Poids (4), 200m, Longueur, Javelot (600), 800m",
            "Heptathlon": "100m haies (84.0), Hauteur, Poids (4), 200m, Longueur, Javelot (600), 800m",
            "Ubs Kids Cup": "60m, Longueur, Balle (200g)",
            "Tétrathlon": "60m, longueur,balle,1000m",
            "Pentathlon 1 U14": "60m,longueur,javelot,hauteur,1000m",
            "Pentathlon 2 U14": "60m,poids,hauteur,javelot,1000m",
            "Hexathlon U16":"100m haies, Longueur, Poids, Hauteur, Disque, 1000m",
            "Pentathlon U16": "80m, Longueur, Poids(3 Femme, 4 Homme), Hauteur, 1000m",
            "Pentathlon U16 (depuis 2024)": "80m haies, Longueur, Poids(3), Hauteur, 600m",
            "Pentathlon U18": "100m, Longueur, Poids(3 Femme, 5 Homme), Hauteur, 1000m",
            "Heptathlon indoor": "60m, Longueur, Poids (7.26), Hauteur, 60m haies (106.7), Perche, 800m",
            "Pentathlon indoor": "60m haies (84.0), Hauteur, Poids (4), Longueur, 800m",
        }
        details = details_disciplines_multiples.get(discipline, "")
        titre = f"{discipline} <span style='color:gray; font-size:0.9em;'>({details})</span>"
        st.markdown(titre, unsafe_allow_html=True)
        
        def format_cell(val, with_none=False):
            if pd.isna(val) or str(val).strip() == "":
                return '<span style="color: #aaa;">None</span>' if with_none else ""
            return str(val)
        
        # ---- CSS (table + tooltip hover PC / tap mobile) ----
        st.markdown("""
        <style>
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        thead {
            background-color: #f8f9fb;
            color: #000000;
        }
        @media (prefers-color-scheme: dark) {
            thead {
                background-color: #1a1c24 !important;
                color: #ffffff !important;
            }
        }
        th {
            padding: 8px 10px;
            text-align: left;
            font-weight: 600;
            color: #9f9b9a;
        }
        td {
            padding: 6px 10px;
            border-bottom: 1px solid #ddd;
        }
        
        /* ===== Tooltip compatible desktop + mobile (sans JS) ===== */
        .info-wrap {
            position: relative;
            display: inline-block;
            margin-left: 6px;
            vertical-align: middle;
        }
        
        .info-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 999px;
            border: 1px solid rgba(0,0,0,0.15);
            background: rgba(255,255,255,0.8);
            cursor: pointer;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
            padding: 0;
            line-height: 1;
        }
        
        @media (prefers-color-scheme: dark) {
            .info-btn {
                border: 1px solid rgba(255,255,255,0.25);
                background: rgba(0,0,0,0.2);
                color: #fff;
            }
        }
        
        .info-btn:focus {
            outline: 2px solid rgba(31,119,180,0.6);
            outline-offset: 2px;
        }
        
        /* Bulle */
        .info-pop {
            display: none;
            position: absolute;
            z-index: 9999;
            top: 110%;
        
            /* 👉 placement intelligent */
            left: 0;
            transform: translateX(0);
        
            /* 👉 une seule ligne */
            white-space: nowrap;
        
            /* 👉 rester dans l’écran */
            max-width: calc(100vw - 24px);
            overflow: hidden;
            text-overflow: ellipsis;
        
            padding: 8px 10px;
            border-radius: 10px;
            background: #ffffff;
            color: #111;
            border: 1px solid rgba(0,0,0,0.15);
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        }
        
        /* Mode sombre */
        @media (prefers-color-scheme: dark) {
            .info-pop {
                background: #151823;
                color: #fff;
                border: 1px solid rgba(255,255,255,0.18);
            }
        }

        @media (prefers-color-scheme: dark) {
            .info-pop {
                background: #151823;
                color: #fff;
                border: 1px solid rgba(255,255,255,0.18);
            }
        }
        
        /* Hover (PC) : on l’active sans media query (plus fiable) */
        .info-wrap:hover .info-pop {
            display: block;
        }
        
        /* Bonus: si la souris est au-dessus du tooltip lui-même, il reste visible */
        .info-wrap:hover,
        .info-wrap:hover .info-pop {
            pointer-events: auto;
        }
        
        /* Mobile + clavier : tap = focus */
        .info-wrap:focus-within .info-pop {
            display: block;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # ---- Construire le tableau HTML (comme avant) ----
        table_html = "<table><thead><tr>"
        headers = ["", "Prénom Nom", "Naissance", "Performance", "Lieu", "Date", "Cat.", "Record"]
        for col in headers:
            table_html += f"<th>{col}</th>"
        table_html += "</tr></thead><tbody>"
        
        for i, (_, row) in enumerate(filtre_affichage.iterrows()):
            table_html += "<tr>"
            table_html += f"<td>{int(row['rang'])}</td>"
            table_html += f"<td>{format_cell(row['Prénom Nom'])}</td>"
            table_html += f"<td>{format_cell(row['Naissance'], with_none=True)}</td>"
        
            details_txt = str(row["Détails"]).replace("\n", " ") if pd.notnull(row["Détails"]) else ""
            details_txt_safe = html_lib.escape(details_txt.strip(), quote=True)
        
            if details_txt_safe:
                icon = f"""
                <span class="info-wrap">
                    <button class="info-btn"
                            type="button"
                            aria-label="Voir les détails"
                            onpointerdown="this.focus();">
                        &#9432;
                    </button>
                    <span class="info-pop" role="tooltip">{details_txt_safe}</span>
                </span>
                """
                perf_html = f"{format_cell(row['Performance'])}{icon}"
            else:
                perf_html = format_cell(row["Performance"])
        
            table_html += f"<td>{perf_html}</td>"
            table_html += f"<td>{format_cell(row['Lieu'], with_none=True)}</td>"
            table_html += f"<td>{format_cell(row['Date'], with_none=True)}</td>"
            table_html += f"<td>{format_cell(row['Cat.'])}</td>"
            table_html += f"<td>{format_cell(row['Record'])}</td>"
            table_html += "</tr>"
        
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
    else:
        df_filtre["record"] = df_filtre["record"].fillna("")

        colonnes = ["rang", "nom", "naissance", "performance_affichee", "lieu", "date", "categorie_affichee", "record", "annee_competition"]
        filtre_affichage = df_filtre[colonnes].rename(columns={
            "nom": "Prénom Nom",
            "naissance": "Naissance",
            "performance_affichee": "Performance",
            "lieu": "Lieu",
            "date": "Date",
            "categorie_affichee": "Cat.",
            "record": "Record",
            "annee_competition": "Année"
        })

        st.markdown("""
        <style>
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        thead {
            background-color: #f8f9fb;
            color: #000000;
        }
        @media (prefers-color-scheme: dark) {
            thead {
                background-color: #1a1c24 !important;
                color: #ffffff !important;
            }
        }
        th {
            padding: 8px 10px;
            text-align: left;
            font-weight: 600;
            color: #9f9b9a;
        }
        td {
            padding: 6px 10px;
            border-bottom: 1px solid #ddd;
        }

        .saison-verte td {
            color: #9f9b9a;
            font-weight: 700;
            background-color: #f8f9fb;
        }
        @media (prefers-color-scheme: dark) {
            .saison-verte td {
                color: #ffffff;
                font-weight: 700;
                background-color: #1b1d26;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        def format_cell(val, with_none=False):
            if pd.isna(val) or str(val).strip() == "":
                return '<span style="color: #aaa;">None</span>' if with_none else ""
            return str(val)

        html = "<table><thead><tr>"
        headers = ["", "Prénom Nom", "Naissance", "Performance", "Lieu", "Date", "Cat.", "Record"]
        for col in headers:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"

        for _, row in filtre_affichage.iterrows():
            classe_tr = "saison-verte" if row["Année"] == ANNEE_SAISON else ""
            html += f'<tr class="{classe_tr}">'
            html += f"<td>{int(row['rang'])}</td>"
            html += f"<td>{format_cell(row['Prénom Nom'])}</td>"
            html += f"<td>{format_cell(row['Naissance'], with_none=True)}</td>"
            html += f"<td>{format_cell(row['Performance'])}</td>"
            html += f"<td>{format_cell(row['Lieu'], with_none=True)}</td>"
            html += f"<td>{format_cell(row['Date'], with_none=True)}</td>"
            html += f"<td>{format_cell(row['Cat.'])}</td>"
            html += f"<td>{format_cell(row['Record'])}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
        








