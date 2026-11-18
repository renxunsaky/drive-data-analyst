# drive-data-analyst

Agent analyste de données conversationnel construit avec le [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/getting-started/quickstart).
Donnez-lui un lien vers un CSV de votre Google Drive, posez vos questions en langage naturel : il répond
avec des chiffres réels calculés sur vos données, via des outils fixes — sans exécution de code arbitraire.

## Fonctionnement

- **Connexion Google Drive** : le connecteur `google_drive_mcp` est attaché à l'agent ; il agit au nom de
  l'utilisateur de la session (OBO). Chaque utilisateur dialogue donc avec ses propres fichiers Drive.
- **Outils fixes** (`@agents.tool`, appelés directement par le modèle via `model_access="direct"`) :
  | Outil | Rôle |
  | --- | --- |
  | `load_csv_from_drive` | Télécharge le CSV depuis un lien Drive (jusqu'à 10 Mo) et le met en mémoire de session |
  | `load_csv_text` | Repli : charge un CSV transmis en texte (contenu lu via le connecteur Drive quand le fichier n'est pas partagé par lien) |
  | `describe_columns` | Type détecté, manquants, min/max/moyenne/total ou valeurs distinctes par colonne |
  | `filter_rows` | Filtres combinés en ET (14 opérateurs : eq, ne, gt, gte, lt, lte, contains, in, is_empty, …) |
  | `aggregate_by_group` | sum / mean / min / max / count / nunique / median par groupe, avec filtres facultatifs |
  | `preview_rows` | Échantillon de lignes |
- **Pas de code arbitraire** : aucun sandbox ni outil shell ; le modèle ne peut appeler que ces outils.
- **Mémoire de conversation** : les datasets sont stockés par session (`ctx.session.id`), le dataset actif est
  conservé d'une question à l'autre ; la session déployée est durable (elle survit aux redémarrages, limite
  d'inactivité 4 h / durée de vie 24 h par défaut).

## Démarrage local

```bash
uv run --env-file .env agents chat
