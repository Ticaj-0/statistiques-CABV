import streamlit as st
import pandas as pd
import re

ANNEE_SAISON = 2025
# --- Fonction : convertir 'mm:ss.cc' → secondes ---
def convertir_temps_en_secondes(valeur):
    if isinstance(valeur, str) and re.match(r'^\d+:\d+\.\d+$', valeur):
        minutes, reste = valeur.split(':')
        secondes = float(reste)
        return int(minutes) * 60 + secondes
    try:
        return float(valeur)
    except:
        return None
    
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
    """Retourne True si discipline est un saut/lancer (mesure) ou une épreuve en mètres."""
    if pd.isna(discipline_raw):
        return False
    s = str(discipline_raw).strip().lower()

    mots_mesure = [
        "hauteur", "longueur", "perche", "triple",
        "poids", "disque", "javelot", "marteau", "balle"
    ]
    return any(m in s for m in mots_mesure)

def est_discipline_points(discipline_raw: str) -> bool:
    """Décathlon, heptathlon, pentathlon… => points."""
    if pd.isna(discipline_raw):
        return False
    s = str(discipline_raw).strip().lower()
    return any(x in s for x in ["athlon", "tétrathlon", "pentathlon", "heptathlon", "décathlon", "decathlon"])

def norm_txt(s):
    return str(s).strip().lower()

def performance_to_key(val, kind: str):
    """
    kind:
      - "t" => temps (centièmes)
      - "d" => distance/hauteur (centimètres)
      - "p" => points (entier)
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
        except:
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
        except:
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

    # distance type 2m40 / 46m74 / 40m00  => 2.40 / 46.74
    if re.match(r"^\d+(\.\d+)?m\d+$", s):
        try:
            metres = float(s.replace("m", ".", 1))
            return ("d", int(round(metres * 100)))  # cm
        except:
            return None

    # nombre simple (c'est ici que ça cassait le 100m)
    try:
        x = float(s)
        if kind == "t":
            return ("t", int(round(x * 100)))   # centièmes
        if kind == "d":
            return ("d", int(round(x * 100)))   # cm
        if kind == "p":
            return ("p", int(round(x)))         # points
        return None
    except:
        return None

def normaliser_categorie_records(groupe):
    """
    Records:
      - 'U14 W' -> 'u14 femme'
      - 'U14 M' -> 'u14 homme'
      - 'Homme' -> 'adulte homme'
      - 'Femmes' -> 'adulte femme'
    """
    s = str(groupe).strip().lower()

    s = s.replace("femmes", "femme").replace("hommes", "homme")
    s = re.sub(r"\bu(\d+)\s*w\b", r"u\1 femme", s)
    s = re.sub(r"\bu(\d+)\s*m\b", r"u\1 homme", s)

    # Adultes
    if s in ["homme", "femme"]:
        s = "adulte " + s

    return s

def normaliser_discipline(d):
    if pd.isna(d):
        return ""

    s = str(d).strip().lower()
    s = s.replace("\u00a0", " ").replace("×", "x")
    s = s.replace(",", ".")
    s = s.replace("’", "'")  # apostrophe typographique -> '

    # enlever parenthèses/crochets
    s = re.sub(r"[\(\)\[\]]", " ", s)

    # ✅ supprimer séparateurs de milliers ENTRE chiffres: 1'000 -> 1000
    s = re.sub(r"(?<=\d)'\s*(?=\d)", "", s)

    # ✅ normaliser relais : "4 x 100" -> "4x100"
    s = re.sub(r"\s*x\s*", "x", s)

    # ✅ HAIES : supprimer la hauteur finale (106.7, 91.4, ...)
    s = re.sub(r"(haies)\s*\d+(\.\d+)?\s*$", r"\1", s)

    # ✅ IMPORTANT : "gr." / "gr" -> "g" (et enlève le point qui casse les matchs)
    s = re.sub(r"\bgr\.?", "g", s)   # marche pour "gr." et "gr"
    s = s.replace("g.", "g")         # sécurité si un point traîne encore

    # supprimer tous les espaces
    s = re.sub(r"\s+", "", s)

    return s

# --- Fonction : calcul de la catégorie selon l'âge ---
def calcul_categorie(age):
    if age >= 20:
        return 'Adulte'
    elif age >= 18:
        return 'U20'
    elif age >= 16:
        return 'U18'
    elif age >= 14:
        return 'U16'
    elif age >= 12:
        return 'U14'
    elif age >= 10:
        return 'U12'
    elif age >= 8:
        return 'U10'
    else:
        return 'Moins de 8 ans'

def compter_decimales_temps(s: str) -> int:
    """
    Retourne le nombre de décimales disponibles sur la partie secondes.
    Ex: "11.2" -> 1 ; "11.25" -> 2 ; "1:34.2" -> 1 ; "1:34.29" -> 2 ; "1:34" -> 0
    """
    if s is None:
        return 0
    s = str(s).strip()

    # enlever le suffixe manuel si présent
    if s.lower().endswith("m"):
        s = s[:-1].strip()

    # normaliser séparateurs pour analyse
    s = s.replace("'", ":").replace('"', ".").replace(",", ".")

    # prendre la partie secondes (après ':', sinon tout)
    sec_part = s.split(":")[-1]

    if "." in sec_part:
        return len(sec_part.split(".")[1])
    return 0


def ajuster_temps_manuel_pour_classement(perf_seconds: float, nb_decimales: int, manuel: bool) -> float:
    """
    Si manuel, on prend la borne haute de l'intervalle de précision :
      - 1 décimale => +0.09  (x.xm -> x.(x+0.09))
      - 0 décimale => +0.99
      - >=2 décimales => +0.00 (déjà au centième ou mieux)
    """
    if perf_seconds is None or pd.isna(perf_seconds):
        return perf_seconds
    if not manuel:
        return perf_seconds

    if nb_decimales == 0:
        return perf_seconds + 0.99
    if nb_decimales == 1:
        return perf_seconds + 0.09
    return perf_seconds

def surligner_derniere_saison(row):
    """
    Surligne la ligne si la date appartient à l'année de la dernière saison.
    """
    try:
        if row.get('annee_competition') == ANNEE_SAISON:
            return ['background-color: #e6f2ff'] * len(row)  # bleu clair
    except:
        pass
    return [''] * len(row)

# --- Mise en cache du chargement et traitement initial des données ---
@st.cache_data
def charger_et_preparer_donnees():
    df = pd.read_excel("Statistiques_2025.xlsx", skiprows=0)

    # IMPORTANT: renommer les colonnes tout de suite
    df.columns = ['discipline', 'performance', 'nom', 'nationalite', 'naissance', 'lieu',
                  'date', 'categorie_fichier', 'sexe', 'record', 'multiple', 'salle']

    df['sexe'] = df['sexe'].replace({'H': 'Homme', 'F': 'Femme'})
    df['performance_brute'] = df['performance']

    df = df.dropna(subset=['discipline', 'performance', 'nom'])

    df['performance_sans_etoile'] = (
        df['performance_brute'].astype(str).str.replace("*", "", regex=False).str.strip()
    )

    # manuel = suffixe "m" à la fin
    df['manuel'] = df['performance_sans_etoile'].astype(str).str.lower().str.endswith('m')
    df['performance_sans_m'] = df['performance_sans_etoile'].astype(str).str[:-1].where(
        df['manuel'], df['performance_sans_etoile']
    ).astype(str).str.strip()

    # conversion numérique (pour classement)
    df['performance'] = df['performance_sans_m'].apply(convertir_temps_en_secondes)
    df['performance'] = pd.to_numeric(df['performance'], errors='coerce')

    df['decimales_temps'] = df['performance_sans_etoile'].apply(compter_decimales_temps)
    df['performance_classement'] = df.apply(
        lambda r: ajuster_temps_manuel_pour_classement(r['performance'], r['decimales_temps'], r['manuel']),
        axis=1
    )

    def extraire_annee(val):
        if pd.isnull(val):
            return None
        if isinstance(val, (int, float)) and 1900 <= int(val) <= 2100:
            return int(val)
        val_str = str(val).strip()
        if re.match(r'^\d{4}$', val_str):
            return int(val_str)
        try:
            return pd.to_datetime(val_str, errors='coerce').year
        except:
            return None

    df['annee_competition'] = df['date'].apply(extraire_annee)
    df['annee_naissance'] = df['naissance'].apply(extraire_annee)
    df['age'] = df['annee_competition'] - df['annee_naissance']

    df['categorie_calculee'] = df['age'].apply(lambda x: calcul_categorie(x) if pd.notnull(x) else None)
    df['categorie'] = df['categorie_calculee']
    df.loc[df['categorie'].isnull(), 'categorie'] = df.loc[df['categorie'].isnull(), 'categorie_fichier']
    df['categorie'] = df['categorie'].fillna('Adulte')

    df['categorie_sexe'] = df['categorie'] + ' ' + df['sexe']
    df['salle'] = df['salle'].fillna('')

    return df
def categories_candidates_pour_records(cat_norm: str):
    """
    Retourne une liste de catégories possibles à tester dans le lookup records.
    Exemple:
      - 'u20 femme' -> ['u20 femme', 'adulte femme']
      - 'u20 homme' -> ['u20 homme', 'adulte homme']  (si même logique)
      - 'u18 femme' -> ['u18 femme'] (par défaut)
    """
    cat_norm = str(cat_norm).strip().lower()

    # fallback U20 W -> Femmes (records)
    if cat_norm == "u20 femme":
        return ["u20 femme", "adulte femme"]

    return [cat_norm]

def appliquer_records_outdoor_fast(df: pd.DataFrame, df_records_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # On n’affecte que Outdoor
    salle_norm = df["salle"].astype(str).str.strip().str.lower()
    mask_out = salle_norm.eq("outdoor")
    df["record"] = df["record"].fillna("")
    df.loc[~mask_out, "record"] = ""  # indoor => pas de records

    if mask_out.sum() == 0 or df_records_lookup.empty:
        return df

    sub = df.loc[mask_out].copy()

    # --- Normalisations nécessaires au match ---
    sub["discipline_norm"] = sub["discipline"].apply(normaliser_discipline)
    sub["cat_norm"] = (sub["categorie"].astype(str) + " " + sub["sexe"].astype(str)).apply(norm_txt)

    # --- kind: calculé une seule fois par discipline (rapide) ---
    # mapping discipline -> kind ("p"/"d"/"t")
    unique_disc = sub["discipline"].dropna().unique().tolist()
    kind_map = {}
    for d in unique_disc:
        if est_discipline_points(d):
            kind_map[d] = "p"
        elif est_discipline_mesure(d):
            kind_map[d] = "d"
        else:
            kind_map[d] = "t"
    sub["kind"] = sub["discipline"].map(kind_map).fillna("t")

    # --- perf_key sans apply(axis=1) ---
    # zip list comprehension = nettement plus rapide
    sub["perf_key"] = [
        performance_to_key(v, k) for v, k in zip(sub["performance_sans_m"].tolist(), sub["kind"].tolist())
    ]

    # --- Gestion "U20 femme" = fallback vers adulte femme (records) ---
    # On crée une table "candidats" pour le merge, avec priorité
    sub = sub.reset_index(drop=False).rename(columns={"index": "_idx"})
    c1 = sub[["_idx", "discipline_norm", "cat_norm", "perf_key"]].copy()
    c1["prio"] = 0

    mask_u20f = c1["cat_norm"].eq("u20 femme")
    c2 = c1.loc[mask_u20f].copy()
    c2["cat_norm"] = "adulte femme"
    c2["prio"] = 1

    candidats = pd.concat([c1, c2], ignore_index=True)

    # --- Merge records ---
    merged = candidats.merge(
        df_records_lookup,
        how="left",
        left_on=["discipline_norm", "cat_norm", "perf_key"],
        right_on=["discipline_norm", "cat_norm", "perf_key"],
    )

    # garder uniquement les matchs
    merged = merged.dropna(subset=["Rang", "Groupe"])
    if merged.empty:
        df.loc[mask_out, "record"] = ""
        return df

    # Si un athlète match sur 2 catégories (u20 femme et adulte femme), prendre prio 0 (u20) d’abord
    merged = merged.sort_values(["_idx", "prio"]).drop_duplicates(subset=["_idx"], keep="first")

    # texte record (vectorisé)
    rang = merged["Rang"].astype(int)
    groupe = merged["Groupe"].astype(str)

    # texte record (vectorisé + règle catégorie)
    rang = merged["Rang"].astype(int)
    groupe = merged["Groupe"].astype(str)

    # normalisation pour savoir si adulte homme/femme
    groupe_norm = groupe.apply(normaliser_categorie_records)

    # --- Cas général : MP valaisanne ---
    merged["record_txt"] = rang.astype(str) + "e MP VS"

    # ajouter la catégorie uniquement si ≠ adulte homme/femme
    mask_add_cat_mp = (~groupe_norm.isin(["adulte homme", "adulte femme"])) & (~rang.eq(1))
    merged.loc[mask_add_cat_mp, "record_txt"] = (merged.loc[mask_add_cat_mp, "record_txt"] + " " + groupe[mask_add_cat_mp])

    # --- Cas record valaisan ---
    merged.loc[rang.eq(1), "record_txt"] = "Record VS"

    # ajouter la catégorie pour record valaisan UNIQUEMENT si ≠ adulte homme/femme
    mask_add_cat_record = rang.eq(1) & (~groupe_norm.isin(["adulte homme", "adulte femme"]))
    merged.loc[mask_add_cat_record, "record_txt"] = ("Record VS " + groupe[mask_add_cat_record])

    # réinjecter dans sub puis df
    sub = sub.merge(merged[["_idx", "record_txt"]], on="_idx", how="left")
    sub["record"] = sub["record_txt"].fillna("")

    df.loc[mask_out, "record"] = sub.set_index("_idx")["record"].values
    return df

@st.cache_data
def charger_records_df():
    df_records = pd.read_excel("records_valaisans_2025.xlsx")

    # Normalisation
    df_records["discipline_norm"] = df_records["Discipline"].apply(normaliser_discipline)
    df_records["cat_norm"] = df_records["Groupe"].apply(normaliser_categorie_records)

    # perf_key (Series.apply OK ici: records << base club en volume)
    df_records["perf_key"] = df_records["Performance"].apply(
        lambda v: performance_to_key(v, kind_records(v))
    )

    # garder seulement les lignes utiles
    df_records = df_records.dropna(subset=["Rang", "discipline_norm", "cat_norm", "perf_key"])

    # typage
    df_records["Rang"] = df_records["Rang"].astype(int)
    df_records["Groupe"] = df_records["Groupe"].astype(str).str.strip()

    # On garde une table de lookup pour merge
    return df_records[["discipline_norm", "cat_norm", "perf_key", "Rang", "Groupe"]].copy()

@st.cache_data
def chargement_des_donnees():
    df_base = charger_et_preparer_donnees()
    df_records_lookup = charger_records_df()
    df_base = appliquer_records_outdoor_fast(df_base, df_records_lookup)
    return df_base

@st.cache_data
def charger_limites():
    df_limites = pd.read_excel("limites.xlsx")
    df_limites["Saison"] = df_limites["Saison"].astype(str).str.strip()
    df_limites["Discipline"] = df_limites["Discipline"].astype(str).str.strip()
    df_limites["Categorie"] = df_limites["Categorie"].astype(str).str.strip()
    df_limites["Sexe"] = df_limites["Sexe"].astype(str).str.strip()
    return df_limites
@st.cache_data
def charger_doublons_valides():
    try:
        return pd.read_excel("doublons_valides.xlsx")
    except FileNotFoundError:
        return pd.DataFrame()

with st.spinner("Chargement des statistiques…"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo_CABV.png", width=180)

    df = chargement_des_donnees()
    df_limites = charger_limites()
    df_doublons_valides = charger_doublons_valides()

# --- Interface Utilisateur ---
st.title(f"Statistiques CABV Martigny {ANNEE_SAISON}")
st.subheader("Filtres")

type_salle = st.selectbox("Saison", ["Outdoor", "Indoor"])
categorie_base_order = ['Homme', 'Femme', 'U20', 'U18', 'U16', 'U14', 'U12', 'U10']
sexe_order = ['Homme', 'Femme']
categorie_order = ['Homme', 'Femme'] + [f"{cat} {sexe}" for cat in categorie_base_order if cat not in ['Homme', 'Femme'] for sexe in sexe_order]
categorie_selection = st.selectbox("Choisir une catégorie", categorie_order)

# --- Mise à jour des disciplines dynamiquement ---
categorie_base = categorie_selection.split()[0] if ' ' in categorie_selection else categorie_selection
sexe_sel = categorie_selection.split()[-1]
categorie_hierarchie = ['U10', 'U12', 'U14', 'U16', 'U18', 'U20', 'Adulte', 'Homme', 'Femme']
indice = categorie_hierarchie.index(categorie_base)
categories_autorisees = categorie_hierarchie[:indice + 1]
categories_autorisees_full = [f"{cat} {sexe_sel}" for cat in categories_autorisees if cat not in ['Homme', 'Femme']]
if categorie_base in ['Homme', 'Femme']:
    categories_autorisees_full.append(sexe_sel)

df['salle'] = df['salle'].fillna('')
df_temp = df[df['salle'].str.lower() == type_salle.lower()]
discipline_possibles = df_temp[df_temp['categorie_sexe'].isin(categories_autorisees_full)]['discipline'].unique().tolist()
ordre_discipline = [
    '50m', '60m', '80m', '100m', '200m', '300m', '400m', '600m', '800m', '1000m', '1500m', '2000m', '3000m', '5000m', '10000m', '50m haies 76.2', '50m haies 106.7', '60m haies 60.0', '60m haies 68.0', '60m haies 76.2',
    '60m haies 106.7', '80m haies 76.2', '100m haies 76.2', '100m haies 84.0', '110m haies 91.4', '110m haies 99.1', '110m haies 106.7', '300m haies 76.2', '300m haies 84.0', '300m haies 91.4',
    '400m haies 76.2', '400m haies 84.0', '400m haies 91.4', '2000m steeple', '3000m steeple', 'Hauteur', 'Longueur', 'Longueur [zone]', 'Perche', 'Triple', 'Poids 2.5kg',
    'Poids 3kg', 'Poids 4kg', 'Poids 5kg', 'Poids 6kg', 'Poids 7.26kg', 'Disque 0.75kg', 'Disque 1kg', 'Disque 1.5kg', 'Disque 1.75kg',
    'Disque 2kg', 'Javelot 400g', 'Javelot 500g', 'Javelot 600g', 'Javelot 700g', 'Javelot 800g', 'Marteau 3kg', 'Marteau 4kg', 'Marteau 5kg',
    'Marteau 6kg', 'Marteau 7.26kg', 'Balle 200g', 'Tétrathlon', 'Décathlon U18', 'Décathlon U20', 'Décathlon',
    'Decathlon indoor', 'Heptathlon U18', 'Heptathlon U18 (depuis 2014)', 'Heptathlon U20', 'Heptathlon', 'Heptathlon indoor', 'Ubs Kids Cup', 'Hexathlon U16',
    'Pentathlon 1 U14', 'Pentathlon 2 U14', 'Pentathlon U16', 'Pentathlon U18', 'Pentathlon U16 (depuis 2024)', 'Relai 5x libre', 'Relai 6x libre', '5x80m', '4x100m'
]

discipline_list = sorted(discipline_possibles, key=lambda x: (ordre_discipline.index(x) if x in ordre_discipline else len(ordre_discipline), x))
discipline = st.selectbox("Choisir une discipline", discipline_list)
mode = st.selectbox("Afficher", ["Un seul résultat par athlète", "Tous les résultats"])
# --- Recherche uniquement sur clic du bouton ---
if st.button("Rechercher"):
    df_filtre = df[(df['salle'].str.lower() == type_salle.lower()) &
                   (df['discipline'] == discipline) &
                   (df['categorie_sexe'].isin(categories_autorisees_full))]
    disciplines_m_mesure = [
    'Hauteur', 'Longueur [zone]', 'Longueur', 'Perche', 'Triple',
    'Poids 2.5kg', 'Poids 3kg', 'Poids 4kg', 'Poids 5kg', 'Poids 6kg', 'Poids 7.26kg', 'Disque 0.75kg', 'Disque 1kg', 'Disque 1.5kg', 'Disque 1.75kg',
    'Disque 2kg', 'Javelot 400g', 'Javelot 500g', 'Javelot 600g', 'Javelot 700g', 'Javelot 800g', 'Marteau 3kg', 'Marteau 4kg', 'Marteau 5kg',
    'Marteau 6kg', 'Marteau 7.26kg', 'Balle 200g', 'Tétrathlon', 'Pentathlon 1 U14', 'Pentathlon 2 U14', 'Décathlon U18', 'Décathlon U20', 'Décathlon',
    'Heptathlon U18', 'Heptathlon U18 (depuis 2014)', 'Heptathlon U20', 'Heptathlon', 'Ubs Kids Cup', 'Hexathlon U16', 'Pentathlon U16', 'Pentathlon U18', 'Pentathlon U16 (depuis 2024)']

    disciplines_multiples = ['Tétrathlon', 'Pentathlon 1 U14', 'Pentathlon 2 U14', 'Décathlon U18', 'Décathlon U20', 'Décathlon', 'Heptathlon U18', 'Heptathlon U18 (depuis 2014)', 'Heptathlon U20', 'Heptathlon', 'Ubs Kids Cup', 'Hexathlon U16', 'Pentathlon U16', 'Pentathlon U18', 'Pentathlon U16 (depuis 2024)', 'Heptathlon indoor', 'Pentathlon indoor']
    relais = ['Relai 5x libre', 'Relai 6x libre', '5x80m', '4x100m']

    if mode == "Un seul résultat par athlète":
        if discipline in disciplines_m_mesure:
            df_filtre = df_filtre.sort_values('performance', ascending=False).drop_duplicates('nom', keep='first')
        else:
            df_filtre = df_filtre.sort_values('performance_classement', ascending=True).drop_duplicates('nom', keep='first')

    if discipline in disciplines_m_mesure:
        df_filtre = df_filtre.sort_values('performance', ascending=False).reset_index(drop=True)
    else:
        df_filtre = df_filtre.sort_values('performance_classement', ascending=True).reset_index(drop=True)

    indice_choisi = categorie_hierarchie.index(categorie_base)

    def afficher_categorie(row):
        if pd.notnull(row['categorie_calculee']):
            cat = row['categorie_calculee']
        elif pd.notnull(row['categorie']):
            cat = row['categorie']
        else:
            cat = 'Adulte'

        cat = str(cat).strip().capitalize()

        # Si adulte → on n'affiche rien
        if cat == 'Adulte':
            return ''

        # Sinon on affiche seulement si la catégorie est dans la hiérarchie visible
        if cat in categorie_hierarchie:
            if sexe_sel in ['Homme', 'Femme']:
                if categorie_hierarchie.index(cat) <= categorie_hierarchie.index(sexe_sel):
                    return cat
            else:
                if categorie_hierarchie.index(cat) <= indice_choisi:
                    return cat

        return ''

    df_filtre['categorie_seule'] = df_filtre['categorie_sexe'].apply(lambda x: x.split()[0] if isinstance(x, str) else "")
    df_filtre['categorie_affichee'] = df_filtre.apply(afficher_categorie, axis=1)

    def format_date_flexible(val):
        if pd.isnull(val): return ""
        val_str = str(val).strip()
        if re.match(r'^\d{4}$', val_str):
            return val_str
        try:
            dt = pd.to_datetime(val, errors='coerce')
            return dt.strftime('%d/%m/%Y') if not pd.isnull(dt) else val_str
        except:
            return val_str

    df_filtre['date'] = df_filtre['date'].apply(format_date_flexible)
    df_filtre['naissance'] = df_filtre['naissance'].apply(format_date_flexible)
    df_filtre['date'] = df_filtre['date'].where(df_filtre['date'].notnull() & (df_filtre['date'] != ''), None)
    df_filtre['naissance'] = df_filtre['naissance'].where(df_filtre['naissance'].notnull() & (df_filtre['naissance'] != ''), None)

    def format_performance(val, discipline):
        val_str = str(val)
        suffix = "*" if "*" in val_str else ""
        val_str = val_str.replace("*", "")

        if discipline in disciplines_multiples:
            try:
                return f"{int(float(val_str))}" + suffix
            except:
                return val_str + suffix

        if re.match(r'^\d+:\d+\.\d+$', val_str):
            return val_str + suffix
        try:
            return f"{float(val_str):.2f}" + suffix
        except:
            return val_str + suffix

    df_filtre['performance_affichee'] = df_filtre['performance_brute'].apply(lambda x: format_performance(x, discipline))
    
    # --- Gestion des doublons avec liste de doublons validés ---
    colonnes_dedoublonnage = ['performance_affichee', 'discipline', 'nom', 'naissance', 'lieu', 'date']

    # Sécuriser si df_doublons_valides n'existe pas (fichier absent, etc.)
    df_doublons_valides = df_doublons_valides if 'df_doublons_valides' in globals() else pd.DataFrame()

    def normaliser_date_cle(val):
        """Normalise une date pour comparaison (clé stable). Gère aussi 0000."""
        if pd.isna(val):
            return ""
        s = str(val).strip()

        # uniformiser séparateurs
        s = s.replace(".", "/").replace("-", "/")

        # Cas année 0000 (pandas ne supporte pas year=0)
        if "0000" in s:
            return "0000-01-01"

        # Essayer de parser dd/mm/yyyy (dayfirst)
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return s  # fallback si non parseable
        return dt.strftime("%Y-%m-%d")

    def normaliser_colonnes_cle(df):
        """Nettoie les colonnes de la clé (dates normalisées + strings strip)."""
        df = df.copy()
        for col in colonnes_dedoublonnage:
            if col in ['date', 'naissance']:
                df[col] = df[col].apply(normaliser_date_cle)
            else:
                df[col] = df[col].astype(str).str.strip()
        return df

    # Normalisation des deux côtés AVANT création des clés
    df_filtre_norm = normaliser_colonnes_cle(df_filtre)

    df_doublons_valides_norm = pd.DataFrame()
    cles_valides = set()

    if not df_doublons_valides.empty and all(c in df_doublons_valides.columns for c in colonnes_dedoublonnage):
        df_doublons_valides_norm = normaliser_colonnes_cle(df_doublons_valides)
        cles_valides = set(map(tuple, df_doublons_valides_norm[colonnes_dedoublonnage].values))

    # Clé de groupe (tuple)
    df_filtre_norm['_cle'] = list(map(tuple, df_filtre_norm[colonnes_dedoublonnage].values))

    # Marquer doublons + validation
    df_filtre_norm['est_doublon'] = df_filtre_norm.duplicated(subset=colonnes_dedoublonnage, keep=False)
    df_filtre_norm['groupe_valide'] = df_filtre_norm['_cle'].isin(cles_valides)

    # Rang dans chaque groupe (0,1,2,...)
    df_filtre_norm['_rang'] = df_filtre_norm.groupby('_cle').cumcount()

    # Règles:
    # - groupe validé -> garder max 2 occurrences
    # - groupe non validé -> garder 1 occurrence
    a_garder = (df_filtre_norm['groupe_valide'] & (df_filtre_norm['_rang'] < 2)) | \
            (~df_filtre_norm['groupe_valide'] & (df_filtre_norm['_rang'] == 0))

    # Pour affichage: doublons non validés (ceux qui seront réduits à 1)
    doublons_non_valides = df_filtre[df_filtre_norm['est_doublon'] & ~df_filtre_norm['groupe_valide']].copy()

    # Appliquer le filtre de conservation sur df_filtre (original) via l'index
    avant = len(df_filtre)
    df_filtre = df_filtre.loc[df_filtre_norm.index[a_garder]].copy()
    apres = len(df_filtre)
    # Recalcul du classement après suppression des doublons
    df_filtre = df_filtre.reset_index(drop=True)

    # Hiérarchie "jeunes -> plus âgé"
    HIERARCHIE_LIMITES = ["U10", "U12", "U14", "U16", "U18", "U20", "Adulte"]

    def get_limite(discipline, categorie, genre, saison, df_limites):
        discipline = str(discipline).strip()
        categorie = str(categorie).strip()
        genre = str(genre).strip()
        saison_norm = str(saison).strip().lower()

        df_lim = df_limites.copy()

        # Normalisation légère (évite les bugs de casse/espaces)
        df_lim["Discipline"] = df_lim["Discipline"].astype(str).str.strip()
        df_lim["Categorie"]  = df_lim["Categorie"].astype(str).str.strip()
        df_lim["Sexe"]       = df_lim["Sexe"].astype(str).str.strip()
        df_lim["Saison"]     = df_lim["Saison"].astype(str).str.strip().str.lower()

        # Liste des catégories à tester (cat demandée puis supérieures)
        cats_a_tester = [categorie]
        if categorie in HIERARCHIE_LIMITES:
            idx = HIERARCHIE_LIMITES.index(categorie)
            cats_a_tester += HIERARCHIE_LIMITES[idx+1:]  # catégories supérieures

        for cat_test in cats_a_tester:
            ligne = df_lim[
                (df_lim["Discipline"] == discipline) &
                (df_lim["Categorie"] == cat_test) &
                (df_lim["Sexe"] == genre) &
                (df_lim["Saison"] == saison_norm)
            ]
            if not ligne.empty:
                return ligne.iloc[0]["Limite"], cat_test  # on renvoie aussi la catégorie utilisée

        return None, None
    
    categorie_sel_clean = categorie_selection.split()[0] if ' ' in categorie_selection else categorie_selection
    limite, categorie_limite_utilisee = get_limite(discipline, categorie_sel_clean, sexe_sel, type_salle, df_limites)

    def convertir_limite(limite_brute):
        if limite_brute is None or (isinstance(limite_brute, float) and pd.isna(limite_brute)):
            return None

        s = str(limite_brute).strip().lower()

        # Normalisation caractères
        s = s.replace(",", ".")
        s = s.replace("'", ":").replace('"', ".")

        # --- temps courts : 10"70 -> 10.70
        if re.match(r'^\d+(\.\d+)?$', s):
            return float(s)

        # --- format mm:ss.xx -> 3:13.00
        if re.match(r'^\d+:\d+(\.\d+)?$', s):
            return convertir_temps_en_secondes(s)

        # --- mètres : 40m00 -> 40.00
        if re.match(r'^\d+(\.\d+)?m\d*$', s):
            s = s.replace("m", ".")
            try:
                return float(s)
            except:
                return None

        # --- points : 2000pts
        if re.match(r'^\d+(\.\d+)?pts$', s):
            try:
                return float(s.replace("pts", ""))
            except:
                return None

        return None

    limite_val = convertir_limite(limite)

    # --- Mode mixte: résultats qualifiés uniquement + meilleur des non qualifiés ---
    if mode == "Tous les résultats" and limite_val is not None and not df_filtre.empty:

        # Définir la condition de qualification + sens du "meilleur"
        if discipline in disciplines_m_mesure:
            # sauts/lancers : plus grand = meilleur, qualifié si >= limite
            passe = df_filtre['performance'] >= limite_val
            meilleur_asc = False
        else:
            # courses : on utilise performance_classement (manuel pénalisé)
            passe = df_filtre['performance_classement'] <= limite_val
            meilleur_asc = True

        df_filtre = df_filtre.copy()
        df_filtre['_passe_limite'] = passe

        # Athlètes qui ont au moins un résultat qualifié
        noms_qualifies = df_filtre.groupby('nom')['_passe_limite'].any()
        noms_qualifies = noms_qualifies[noms_qualifies].index

        # 1) Qualifiés: uniquement les lignes qui passent la limite
        df_qualifies = df_filtre[df_filtre['nom'].isin(noms_qualifies) & df_filtre['_passe_limite']]

        # 2) Non qualifiés: garder uniquement leur meilleur résultat
        df_nonqualif = df_filtre[~df_filtre['nom'].isin(noms_qualifies)]
        df_nonqualif_best = (
            df_nonqualif.sort_values('performance_classement', ascending=meilleur_asc)
                        .drop_duplicates('nom', keep='first')
        )

        # Combiner
        df_filtre = pd.concat([df_qualifies, df_nonqualif_best], ignore_index=True)

        # Re-tri final + index 1..N
        df_filtre = df_filtre.sort_values('performance_classement', ascending=meilleur_asc).reset_index(drop=True)

        # Nettoyage
        df_filtre = df_filtre.drop(columns=['_passe_limite'], errors='ignore')

        def texte_complement_discipline(discipline, categorie_selection, type_salle):
            """
            Ajoute un texte complémentaire selon discipline + catégorie + indoor/outdoor.
            """

            # 🔹 Indoor : toujours "(depuis 2006)"
            if type_salle.lower() == "indoor":
                return " (depuis 2006)"

            # 🔹 Outdoor : règles spécifiques U10
            if not categorie_selection.startswith("U10"):
                return ""

            if discipline == "50m":
                return " (depuis 2016)"
            
            if discipline == "600m":
                return " (depuis 2017)"

            return ""

    texte_plus = texte_complement_discipline(discipline, categorie_selection)

    if mode == "Tous les résultats" and limite is not None:
        if discipline in disciplines_m_mesure:
            st.subheader(f"Résultats au dessus de {limite} + Top 200 - {discipline}{texte_plus} - {categorie_selection}")
        else:
            st.subheader(f"Résultats en dessous de {limite} + Top 200 - {discipline}{texte_plus}  - {categorie_selection}")
    else:
        st.subheader(f"{mode} - {discipline}{texte_plus}  - {categorie_selection}")
    
    # --- Classement avec ex-aequo (1,2,3,3,3,6...) ---
    if discipline in disciplines_m_mesure:
        # mesures : plus grand = meilleur
        df_filtre = df_filtre.sort_values('performance', ascending=False).reset_index(drop=True)
        df_filtre['rang'] = df_filtre['performance'].rank(method='min', ascending=False).astype(int)
    else:
        # temps : plus petit = meilleur (et manuel déjà pénalisé via performance_classement)
        df_filtre = df_filtre.sort_values('performance_classement', ascending=True).reset_index(drop=True)
        df_filtre['rang'] = df_filtre['performance_classement'].rank(method='min', ascending=True).astype(int)

    if df_filtre.empty:
        st.warning("Aucun résultat trouvé pour cette sélection.")
    # Si discipline multiple, afficher tableau personnalisé avec tooltip
    elif discipline in disciplines_multiples or discipline in relais:
        # Colonnes utiles
        colonnes = ['rang', 'nom', 'naissance', 'performance_affichee', 'lieu', 'date', 'categorie_affichee', 'record', 'multiple']
        filtre_affichage = df_filtre[colonnes].rename(columns={
            'nom': 'Prénom Nom',
            'naissance': 'Naissance',
            'performance_affichee': 'Performance',
            'lieu': 'Lieu',
            'date': 'Date',
            'categorie_affichee': 'Cat.',
            'record': 'Record',
            'multiple': 'Détails'
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
        details = details_disciplines_multiples.get(discipline, '')
        titre = f"{discipline} <span style='color:gray; font-size:0.9em;'>({details})</span>"

        st.markdown(titre, unsafe_allow_html=True)

        # Affichage HTML avec tooltip + icône info
        def tooltip_html(row):
            perf = row['Performance']
            details = str(row['Détails']).replace("\n", " ")
            if pd.isna(row['Détails']) or row['Détails'] == "":
                return perf
            icon = f'<span title="{details}" style="cursor: help; color: #1f77b4; margin-left:5px;">&#9432;</span>'  # &#9432; = ℹ️
            return f"{perf}{icon}"

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
        /* Style par défaut (mode clair) */
        thead {
            background-color: #f8f9fb;
            color: #000000;
        }

        /* Mode sombre */
        @media (prefers-color-scheme: dark) {
            thead {
                background-color: #1a1c24 !important;  /* gris foncé lisible en mode sombre */
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
        td span[title] {
            cursor: help;
            color: #1f77b4 ;
            margin-left: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Fonctions de formatage conditionnel
        def format_cell(val, with_none=False):
            if pd.isna(val) or str(val).strip() == '':
                return '<span style="color: #aaa;">None</span>' if with_none else ''
            return str(val)

        # Génération du tableau HTML avec classement
        html = "<table><thead><tr>"
        headers = ['', 'Prénom Nom', 'Naissance', 'Performance', 'Lieu', 'Date', 'Cat.', 'Record']
        for col in headers:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"

        for _, row in filtre_affichage.iterrows():
            html += "<tr>"
            html += f"<td>{int(row['rang'])}</td>"  # Rang

            html += f"<td>{format_cell(row['Prénom Nom'])}</td>"
            html += f"<td>{format_cell(row['Naissance'], with_none=True)}</td>"

            # Performance avec tooltip s’il y a des détails
            details = str(row['Détails']).replace("\n", " ") if pd.notnull(row['Détails']) else ""
            if details.strip():
                icon = f'<span title="{details}">&#9432;</span>'
                perf_html = f"{format_cell(row['Performance'])}{icon}"
            else:
                perf_html = format_cell(row['Performance'])
            html += f"<td>{perf_html}</td>"

            html += f"<td>{format_cell(row['Lieu'], with_none=True)}</td>"
            html += f"<td>{format_cell(row['Date'], with_none=True)}</td>"
            html += f"<td>{format_cell(row['Cat.'])}</td>"
            html += f"<td>{format_cell(row['Record'])}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        df_filtre['record'] = df_filtre['record'].fillna('')

        # Colonnes utiles (même rendu que disciplines multiples)
        colonnes = [
            'rang',
            'nom',
            'naissance',
            'performance_affichee',
            'lieu',
            'date',
            'categorie_affichee',
            'record',
            'annee_competition'
        ]

        filtre_affichage = df_filtre[colonnes].rename(columns={
            'nom': 'Prénom Nom',
            'naissance': 'Naissance',
            'performance_affichee': 'Performance',
            'lieu': 'Lieu',
            'date': 'Date',
            'categorie_affichee': 'Cat.',
            'record': 'Record',
            'annee_competition': 'Année'
        })

        # CSS identique aux disciplines multiples + texte vert saison
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

        /* Texte vert pour la saison courante */
        .saison-verte td {
            color: #1b8f3a;
            font-weight: 700;
        }

        /* Vert lisible en mode sombre */
        @media (prefers-color-scheme: dark) {
            .saison-verte td {
                color: #4cd964;
                font-weight: 700;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        def format_cell(val, with_none=False):
            if pd.isna(val) or str(val).strip() == '':
                return '<span style="color: #aaa;">None</span>' if with_none else ''
            return str(val)

        # Génération HTML
        html = "<table><thead><tr>"
        headers = ['', 'Prénom Nom', 'Naissance', 'Performance', 'Lieu', 'Date', 'Cat.', 'Record']
        for col in headers:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"

        for i, (_, row) in enumerate(filtre_affichage.iterrows(), start=1):
            classe_tr = "saison-verte" if row['Année'] == ANNEE_SAISON else ""
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