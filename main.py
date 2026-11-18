"""Agent analyste de données conversationnel : CSV depuis Google Drive, outils d'analyse fixes."""

import csv
import io
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from mistralai import agents

MAX_CSV_BYTES = 10 * 1024 * 1024
PREVIEW_ROW_LIMIT = 50
AGGREGATE_GROUP_LIMIT = 200
FILTER_ROW_LIMIT = 50
NUMERIC_RATIO = 0.8


class Dataset:
    def __init__(self, name: str, file_name: str, rows: list[dict[str, str]]) -> None:
        self.name = name
        self.file_name = file_name
        self.rows = rows

    @property
    def columns(self) -> list[str]:
        return list(self.rows[0].keys()) if self.rows else []


_STORES: dict[str, dict[str, Dataset]] = {}
_ACTIVE: dict[str, str] = {}


def _store_for(session_id: str) -> dict[str, Dataset]:
    return _STORES.setdefault(session_id, {})


def _get_dataset(session_id: str, dataset_name: str | None) -> Dataset:
    name = dataset_name or _ACTIVE.get(session_id, "default")
    store = _store_for(session_id)
    dataset = store.get(name)
    if dataset is None:
        available = ", ".join(store) or "(aucun)"
        raise agents.ToolError(
            f"Aucun dataset {name!r} chargé dans cette session. Datasets disponibles : {available}. "
            "Rechargez le dernier fichier connu avec load_csv_from_drive ou load_csv_text."
        )
    return dataset


def save_dataset(session_id: str, name: str, file_name: str, text: str) -> Dataset:
    rows = _parse_csv(text)
    dataset = Dataset(name=name, file_name=file_name, rows=rows)
    _store_for(session_id)[name] = dataset
    _ACTIVE[session_id] = name
    return dataset


class FilterCondition(BaseModel):
    column: str = Field(description="Nom exact de la colonne à filtrer.")
    operator: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte",
        "contains", "not_contains", "startswith", "endswith",
        "in", "not_in", "is_empty", "not_empty",
    ] = Field(
        description=(
            "Opérateur de comparaison. eq/ne/gt/gte/lt/lte comparent numériquement quand c'est possible, "
            "sinon sur le texte. in/not_in prennent une liste de valeurs séparées par '|'. "
            "is_empty/not_empty ignorent value."
        )
    )
    value: str = Field(
        default="",
        description="Valeur de comparaison. Pour in/not_in, valeurs séparées par '|'. Ignorée pour is_empty/not_empty.",
    )


def resolve_column(dataset: Dataset, column: str) -> str:
    for existing in dataset.columns:
        if existing.strip().lower() == column.strip().lower():
            return existing
    raise agents.ToolError(
        f"Colonne {column!r} introuvable. Colonnes disponibles : {dataset.columns}"
    )


def _to_number(raw: str) -> float | int | None:
    text = raw.strip().replace(" ", "").replace("\u00a0", "").rstrip("%")
    if not text:
        return None
    neg_parenth = text.startswith("(") and text.endswith(")")
    if neg_parenth:
        text = text[1:-1]
    if "," in text and (text.count(",") > text.count(".")):
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    if neg_parenth:
        number = -number
    if number.is_integer() and "." not in text and "e" not in text.lower():
        return int(number)
    return number


def _match(row: dict[str, str], condition: FilterCondition, column: str) -> bool:
    raw = row.get(column, "")
    op = condition.operator
    value = condition.value
    if op == "is_empty":
        return raw.strip() == ""
    if op == "not_empty":
        return raw.strip() != ""
    cell_number = _to_number(raw)
    value_number = _to_number(value)
    if cell_number is not None and value_number is not None and op in {"eq", "ne", "gt", "gte", "lt", "lte"}:
        return {
            "eq": cell_number == value_number,
            "ne": cell_number != value_number,
            "gt": cell_number > value_number,
            "gte": cell_number >= value_number,
            "lt": cell_number < value_number,
            "lte": cell_number <= value_number,
        }[op]
    if op in {"eq", "ne", "contains", "not_contains", "startswith", "endswith"}:
        return {
            "eq": raw.strip().lower() == value.strip().lower(),
            "ne": raw.strip().lower() != value.strip().lower(),
            "contains": value.strip().lower() in raw.lower(),
            "not_contains": value.strip().lower() not in raw.lower(),
            "startswith": raw.strip().lower().startswith(value.strip().lower()),
            "endswith": raw.strip().lower().endswith(value.strip().lower()),
        }[op]
    if op in {"in", "not_in"}:
        if cell_number is not None:
            options = [n for n in (_to_number(item) for item in value.split("|")) if n is not None]
            if options:
                return (cell_number in options) == (op == "in")
        options = [item.strip().lower() for item in value.split("|")]
        return (raw.strip().lower() in options) == (op == "in")
    return False


def apply_filters(
    dataset: Dataset, filters: list[FilterCondition]
) -> list[dict[str, str]]:
    rows = dataset.rows
    for condition in filters:
        column = resolve_column(dataset, condition.column)
        rows = [row for row in rows if _match(row, condition, column)]
    return rows


def _parse_csv(text: str) -> list[dict[str, str]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        lines = sample.splitlines()
        header_line = lines[0] if lines else ""
        delimiter = ";" if header_line.count(";") > header_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise agents.ToolError("Fichier CSV vide ou sans ligne d'en-tête.")
    rows = []
    for row in reader:
        cleaned = {
            (key or "colonne_vide").strip(): (value if value is not None else "")
            for key, value in row.items()
        }
        rows.append(cleaned)
    if not rows:
        raise agents.ToolError("Fichier CSV sans lignes de données.")
    return rows


_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def extract_drive_file_id(url: str) -> str:
    url = url.strip()
    if re.fullmatch(r"[\w-]{20,}", url):
        return url
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or (parsed.hostname or "") not in _DRIVE_HOSTS:
        raise agents.ToolError(
            "URL Google Drive invalide. Fournissez un lien https://drive.google.com/... ou https://docs.google.com/..."
        )
    match = re.search(r"/d/([\w-]{20,})", parsed.path)
    if match:
        return match.group(1)
    query = urllib.parse.parse_qs(parsed.query)
    if "id" in query:
        return query["id"][0]
    raise agents.ToolError("Impossible d'extraire l'identifiant du fichier depuis ce lien Drive.")


def download_drive_csv(url: str) -> tuple[str, str]:
    file_id = extract_drive_file_id(url)
    candidates = [
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv",
    ]
    last_error = "réponse vide"
    for candidate in candidates:
        try:
            request = urllib.request.Request(
                candidate, headers={"User-Agent": "Mozilla/5.0 (csv-analyst-agent)"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(MAX_CSV_BYTES + 1)
                content_type = response.headers.get("Content-Type", "")
            if len(data) > MAX_CSV_BYTES:
                raise agents.ToolError(
                    f"Fichier trop volumineux (limite {MAX_CSV_BYTES // (1024 * 1024)} Mo)."
                )
            text = data.decode("utf-8-sig", errors="replace")
            if is_plausible_csv(text):
                return text, file_id
            last_error = f"contenu non CSV (type {content_type or 'inconnu'} ; page HTML de connexion probable)"
        except agents.ToolError:
            raise
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
    raise agents.ToolError(
        f"Téléchargement direct du fichier Drive impossible ({last_error}). "
        "Le fichier n'est probablement pas partagé 'accessible par lien'. Utilisez les outils du connecteur "
        "Google Drive pour retrouver le fichier et lire son contenu, puis transmettez ce contenu à load_csv_text."
    )


def is_plausible_csv(text: str) -> bool:
    if "<html" in text[:512].lower() or "<!doctype html" in text[:512].lower():
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 2 and ("," in text or ";" in text or "\t" in text)


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def load_csv_from_drive(
    ctx: agents.ToolCallbackContext,
    /,
    drive_url: Annotated[
        str, Field(description="Lien Google Drive du fichier CSV (https://drive.google.com/... ou https://docs.google.com/...) ou son identifiant de fichier.")
    ],
    dataset_name: str = "default",
) -> dict[str, Any]:
    """Charge un CSV depuis un lien Google Drive. À appeler dès que l'utilisateur donne un lien Drive. Rechargez-le si un outil signale que le dataset est introuvable."""
    session_id = ctx.session.id
    text, file_id = download_drive_csv(drive_url)
    dataset = save_dataset(session_id, dataset_name, file_id, text)
    return {
        "dataset": dataset_name,
        "fichier": f"https://drive.google.com/file/d/{file_id}/view",
        "lignes": len(dataset.rows),
        "colonnes": dataset.columns,
        "note": "Dataset actif pour toute la conversation tant qu'un autre fichier n'est pas chargé.",
    }


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def load_csv_text(
    ctx: agents.ToolCallbackContext,
    /,
    csv_text: Annotated[str, Field(description="Contenu CSV complet (délimité, avec ligne d'en-tête).")],
    file_name: str = "fichier.csv",
    dataset_name: str = "default",
) -> dict[str, Any]:
    """Charge un CSV fourni en texte brut. À utiliser quand le téléchargement direct du Drive échoue et que le contenu du fichier a été lu via les outils du connecteur Google Drive."""
    session_id = ctx.session.id
    if len(csv_text.encode("utf-8")) > MAX_CSV_BYTES:
        raise agents.ToolError(f"Contenu trop volumineux (limite {MAX_CSV_BYTES // (1024 * 1024)} Mo).")
    dataset = save_dataset(session_id, dataset_name, file_name, csv_text)
    return {
        "dataset": dataset_name,
        "fichier": file_name,
        "lignes": len(dataset.rows),
        "colonnes": dataset.columns,
        "note": "Dataset actif pour toute la conversation tant qu'un autre fichier n'est pas chargé.",
    }


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def describe_columns(ctx: agents.ToolCallbackContext, /, dataset_name: str = "") -> dict[str, Any]:
    """Décrire les colonnes du dataset chargé : type détecté, valeurs manquantes, stats numériques (min/max/moyenne/total) ou valeurs distinctes et exemples."""
    session_id = ctx.session.id
    dataset = _get_dataset(session_id, dataset_name or None)
    total = len(dataset.rows)
    per_column: dict[str, Any] = {}
    for column in dataset.columns:
        values = [row.get(column, "") for row in dataset.rows]
        present = [value for value in values if value.strip() != ""]
        numbers = [number for number in (_to_number(value) for value in present) if number is not None]
        numeric_enough = len(numbers) >= max(1, int(NUMERIC_RATIO * len(present))) if present else True
        info: dict[str, Any] = {
            "type": "numérique" if numeric_enough else "texte",
            "valeurs_manquantes": total - len(present),
        }
        if numeric_enough and numbers:
            info.update(
                min=min(numbers),
                max=max(numbers),
                moyenne=round(sum(float(n) for n in numbers) / len(numbers), 6),
                total=round(sum(float(n) for n in numbers), 6),
            )
        elif present:
            distinct = sorted({value.strip() for value in present})
            info.update(valeurs_distinctes=len(distinct), exemples=distinct[:8])
        else:
            info.update(valeurs_distinctes=0, exemples=[])
        per_column[column] = info
    return {"dataset": dataset.name, "lignes": total, "colonnes": per_column}


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    filters: Annotated[list[FilterCondition], Field(description="Une ou plusieurs conditions, toutes combinées en ET.")],
    dataset_name: str = "",
    limit: int = FILTER_ROW_LIMIT,
) -> dict[str, Any]:
    """Filtrer les lignes du dataset chargé selon des conditions ; retourne le total de lignes correspondantes et un échantillon."""
    session_id = ctx.session.id
    dataset = _get_dataset(session_id, dataset_name or None)
    matched = apply_filters(dataset, filters)
    sample_limit = max(1, min(limit, FILTER_ROW_LIMIT))
    return {
        "dataset": dataset.name,
        "lignes_total": len(dataset.rows),
        "lignes_correspondantes": len(matched),
        "echantillon": matched[:sample_limit],
        "tronque": len(matched) > sample_limit,
    }


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def aggregate_by_group(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[list[str], Field(description="Colonne(s) de regroupement.")],
    agg: Literal["sum", "mean", "min", "max", "count", "nunique", "median"],
    value_column: str = "",
    filters: Annotated[list[FilterCondition], Field(description="Filtres facultatifs appliqués avant l'agrégation (combinés en ET).")] = [],
    dataset_name: str = "",
    sort: Literal["desc", "asc", "none"] = "desc",
    limit: int = AGGREGATE_GROUP_LIMIT,
) -> dict[str, Any]:
    """Agréger par groupe la colonne numérique indiquée (somme, moyenne, min, max, count, nunique, median), avec filtres facultatifs. value_column est requis sauf pour 'count'."""
    session_id = ctx.session.id
    dataset = _get_dataset(session_id, dataset_name or None)
    rows = apply_filters(dataset, filters) if filters else dataset.rows
    keys = [resolve_column(dataset, column) for column in group_by]
    if agg != "count":
        if not value_column:
            raise agents.ToolError(f"L'agrégation {agg!r} nécessite value_column.")
        value_col = resolve_column(dataset, value_column)
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(key, "") for key in keys)].append(row)
    results: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        entry: dict[str, Any] = {name: value for name, value in zip(keys, key)}
        entry["nombre_lignes"] = len(bucket)
        if agg == "count":
            entry["count"] = len(bucket)
            results.append(entry)
            continue
        numbers = [n for n in (_to_number(row.get(value_col, "")) for row in bucket) if n is not None]
        if not numbers:
            entry[agg] = None
        elif agg == "sum":
            entry[agg] = round(sum(float(n) for n in numbers), 6)
        elif agg == "mean":
            entry[agg] = round(sum(float(n) for n in numbers) / len(numbers), 6)
        elif agg in {"min", "max"}:
            entry[agg] = (min if agg == "min" else max)(numbers)
        elif agg == "median":
            ordered = sorted(numbers)
            mid = len(ordered) // 2
            entry[agg] = (
                ordered[mid] if len(ordered) % 2 else round((ordered[mid - 1] + ordered[mid]) / 2, 6)
            )
        elif agg == "nunique":
            entry[agg] = len({row.get(value_col, "").strip() for row in bucket})
        results.append(entry)
    if sort in {"asc", "desc"} and results:
        sort_key = "count" if agg == "count" else agg
        results.sort(
            key=lambda entry: (entry.get(sort_key) is None, entry.get(sort_key) if entry.get(sort_key) is not None else 0),
            reverse=(sort == "desc"),
        )
    group_limit = max(1, min(limit, AGGREGATE_GROUP_LIMIT))
    return {
        "dataset": dataset.name,
        "lignes_avant_agregation": len(rows),
        "groupes": len(results),
        "resultat": results[:group_limit],
        "tronque": len(results) > group_limit,
    }


@agents.tool(model_access="direct", annotations=agents.ToolAnnotations(read_only_hint=True))
async def preview_rows(
    ctx: agents.ToolCallbackContext,
    /,
    n: int = 5,
    offset: int = 0,
    dataset_name: str = "",
) -> dict[str, Any]:
    """Afficher un échantillon de lignes du dataset chargé, pour vérifier les données ou la structure."""
    session_id = ctx.session.id
    dataset = _get_dataset(session_id, dataset_name or None)
    size = max(1, min(n, PREVIEW_ROW_LIMIT))
    return {
        "dataset": dataset.name,
        "lignes_total": len(dataset.rows),
        "offset": offset,
        "lignes": dataset.rows[offset:offset + size],
    }


agent = agents.Agent(
    model="mistral-large-4",
    name="drive-data-analyst",
    instructions=(
        "Tu es un analyste de données conversationnel. L'utilisateur te donne un lien vers un fichier CSV "
        "de son Google Drive et te pose des questions en langage naturel. Tu réponds uniquement avec des "
        "chiffres réels issus de ses données, jamais avec des estimations inventées.\n\n"
        "Mode opératoire :\n"
        "- À la réception d'un lien Drive, appelle load_csv_from_drive. Si le téléchargement direct échoue "
        "(fichier privé non partagé par lien), utilise les outils du connecteur Google Drive pour retrouver le "
        "fichier et lire son contenu, puis transmets ce contenu à load_csv_text. Explique à l'utilisateur ce qui "
        "s'est passé le cas échéant.\n"
        "- Après un premier chargement, appelle describe_columns pour ancrer la structure des données.\n"
        "- Pour répondre aux questions, utilise uniquement les outils fixes : describe_columns, filter_rows, "
        "aggregate_by_group, preview_rows. Tu ne peux pas exécuter de code arbitraire ; ne prétends jamais le faire.\n"
        "- Les filtres des outils sont combinés en ET ; pour une négation utilise ne, not_contains ou not_in.\n"
        "- Mémoire de conversation : le dataset chargé reste actif, avec son nom et sa structure, tant que "
        "l'utilisateur n'en charge pas un autre. Les questions de suivi se réfèrent implicitement à ce dataset et "
        "aux filtres déjà discutés : n'exige jamais de l'utilisateur qu'il se réexplique. Si un outil signale que "
        "le dataset est introuvable, recharge-le avec le dernier lien ou contenu connu avant de répondre.\n"
        "- Cite les chiffres exacts renvoyés par les outils, précise le nombre de lignes sur lesquels ils sont "
        "calculés et signale explicitement les résultats tronqués.\n"
        "- Si une question est réellement ambiguë (colonne, unité, périmètre temporel), pose une courte question "
        "de clarification ; sinon réponds directement.\n"
        "- Réponds dans la langue de l'utilisateur."
    ),
    connectors=[agents.connector("google_drive_mcp")],
    tools=[load_csv_from_drive, load_csv_text, describe_columns, filter_rows, aggregate_by_group, preview_rows],
)
