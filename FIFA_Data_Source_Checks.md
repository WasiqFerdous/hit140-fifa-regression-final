# Source checks for the FIFA regression datasets

Checked on 13 October 2026.

## Dataset source identified

The strongest source match is [MD Mominul Islam's FIFA World Cup 2026 Dataset](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset), also distributed on [Kaggle](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset).

The file names and 60-column feature schema match. All 48 team names, FIFA ranks and Elo values were compared with the repository generator table and match the supplied feature file. This establishes consistency with that source, not a byte-for-byte match of the complete CSV download. The exact downloaded version and retrieval date remain unconfirmed.

## Preparation details found

- [Feature-generation code](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset/blob/main/generate_prediction_features.py) sets fixed values when a team has no tournament history. It computes subsequent averages from earlier matches, uses 11 June 2026 for squad age and reads Elo from the static team table.
- [Dataset generator](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset/blob/main/generate_dataset.py) provides that team table. Its valuation code can use supplied prices, selected star-player values or rank-based estimates.

The first-match values therefore have a code-level explanation: prior goals 0, shots on target 4, possession 50%, saves 3 and rest 10 days are defaults, not measured prior form. Statistical justification for these prior values remains a limitation.

Squad value is an estimated proxy; the supplied valuations have not all been verified as Transfermarkt totals. The generator labels star prices as a June 2026 snapshot; an exact pre-kickoff valuation date is not established.

## Independent checks

| Check | Outcome | Evidence |
|---|---|---|
| Mexico versus South Africa | Supplied 2-0 score corroborated | [FIFA tournament standings](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/final-tournament-standings) |
| France versus England | Supplied 4-6 score corroborated | [England Football bronze-match report](https://www.englandfootball.com/articles/2026/Jul/18/england-men-achieve-best-world-cup-finish-since-1966-20261807) |
| Spain versus Argentina | Supplied 1-0 score after extra time corroborated | [FIFA final report](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/spain-argentina-final-report-highlights) |
| Ten FIFA ranks | All ten checked values match the 11 June release | [FIFA ranking release](https://inside.fifa.com/fifa-rankings/world-ranking/men/news/argentina-reclaim-top-spot-mens-world-ranking-fifa-coca-cola) |

Checked FIFA positions: Argentina 1, Spain 2, France 3, Morocco 7, Netherlands 8, Mexico 14, Senegal 15, Uruguay 16, USA 17 and IR Iran 20. The release explicitly precedes the opening match.

All 104 scores agree internally between the supplied files. Only the three outcomes above were independently checked in public sources. The repository team table comparison covers 48 ranks and Elo entries; it does not independently authenticate the underlying Elo observations.

## Remaining evidence

- The team's exact download link, version and retrieval date.
- The historical provider and date for static Elo values.
- Exact dates and measured versus estimated status of squad valuations.
- Underlying event records or export metadata for shots, possession and saves.
- A rationale or sensitivity check for fixed initial values.

The source data paper contains an illustrative opener sample inconsistent with the current generator and the supplied score. The checks above use the actual supplied records and corroborated results.
