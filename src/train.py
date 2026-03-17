import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier

virtual_path = Path("./march-madness-26")
base_path = str(virtual_path.resolve())

seeds = pd.read_csv(base_path + "/data/MNCAATourneySeeds.csv")
regular = pd.read_csv(base_path + "/data/MRegularSeasonCompactResults.csv")
tourney = pd.read_csv(base_path + "/data/MNCAATourneyCompactResults.csv")
sample = pd.read_csv(base_path + "/data/SampleSubmissionStage1.csv")
teams = pd.read_csv(base_path + "/data/MTeams.csv")

regular = regular.sort_values(["Season", "DayNum"])

late_wins = regular[["Season","WTeamID"]].rename(columns={"WTeamID":"TeamID"})
late_wins["Win"] = 1
late_losses = regular[["Season","LTeamID"]].rename(columns={"LTeamID":"TeamID"})
late_losses["Win"] = 0
late_games = pd.concat([late_wins, late_losses]).sort_values(["Season","TeamID"])

late_form = (
    late_games.groupby(["Season", "TeamID"])
    .tail(10)
    .groupby(["Season", "TeamID"])["Win"]
    .mean()
    .reset_index()
    .rename(columns={"Win": "LateWinPct"})
)

seeds["SeedNum"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
seeds = seeds[["Season", "TeamID", "SeedNum"]]

wins = regular[["Season", "WTeamID", "WScore", "LScore"]].copy()
wins.columns = ["Season", "TeamID", "PointsFor", "PointsAgainst"]
wins["Win"] = 1

losses = regular[["Season", "LTeamID", "LScore", "WScore"]].copy()
losses.columns = ["Season", "TeamID", "PointsFor", "PointsAgainst"]
losses["Win"] = 0

team_games = pd.concat([wins, losses], ignore_index=True)
team_games["ScoreMargin"] = team_games["PointsFor"] - team_games["PointsAgainst"]

team_stats = (
    team_games
    .groupby(["Season", "TeamID"])
    .agg(
        WinPct=("Win", "mean"),
        AvgPointsFor=("PointsFor", "mean"),
        AvgPointsAgainst=("PointsAgainst", "mean"),
        AvgScoreMargin=("ScoreMargin", "mean"),
        GamesPlayed=("Win", "count"),
    )
    .reset_index()
)

team_stats = team_stats.merge(seeds, on=["Season", "TeamID"], how="left")

team_stats["SeedNum"] = team_stats["SeedNum"].fillna(20)


def build_matchup_rows(df, label_for_winner_first=True):
    if label_for_winner_first:
        left = df[["Season", "WTeamID", "LTeamID"]].copy()
        left.columns = ["Season", "Team1", "Team2"]
        left["Label"] = 1
    else:
        left = df[["Season", "LTeamID", "WTeamID"]].copy()
        left.columns = ["Season", "Team1", "Team2"]
        left["Label"] = 0
    return left


def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def update_elo(r_a, r_b, result_a, k=20):
    expected_a = expected_score(r_a, r_b)

    n_a = r_a + k * (result_a - expected_a)
    n_b = r_b + k * ((1 - result_a) - (1 - expected_a))

    return n_a, n_b


def get_starting_elo(team):
    prior = prev_elo.get(team, 1500)
    return 1500 + CARRY_OVER * (prior - 1500)


# ELO - based on team strength
elo = {}

for _, game in regular.iterrows():

    season = game["Season"]
    team_a = game["WTeamID"]
    team_b = game["LTeamID"]

    key_a = (season, team_a)
    key_b = (season, team_b)

    CARRY_OVER = 0.5

    prev_elo = {}
    for (season, team), rating in elo.items():
        prev_elo[team] = rating  # save end-of-season ratings

    rating_a = elo.get(key_a, get_starting_elo(team_a))
    rating_b = elo.get(key_b, get_starting_elo(team_a))

    home = game.get("WLoc", "N")
    k_value = 15 if home == "H" else 25 if home == "A" else 20
    new_a, new_b = update_elo(rating_a, rating_b, result_a=1, k=k_value)

    elo[key_a] = new_a
    elo[key_b] = new_b

elo_rows = []

for (season, team), rating in elo.items():
    elo_rows.append({
        "Season": season,
        "TeamID": team,
        "EloRating": rating
    })

elo_df = pd.DataFrame(elo_rows)

team_stats = team_stats.merge(
    elo_df,
    on=["Season", "TeamID"],
    how="left"
)

team_stats = team_stats.merge(late_form, on=["Season", "TeamID"], how="left")
team_stats["LateWinPct"] = team_stats["LateWinPct"].fillna(team_stats["WinPct"])

opp_wins = wins.rename(columns={"TeamID": "OppID"})
opp_losses = losses.rename(columns={"TeamID": "OppID"})
opp_games = pd.concat([opp_wins, opp_losses])

base_winpct = team_games.groupby(["Season", "TeamID"])["Win"].mean().reset_index()
base_winpct.columns = ["Season", "OppID", "OppWinPct"]

opp_for_winners = regular[["Season", "WTeamID", "LTeamID"]].copy()
opp_for_winners.columns = ["Season", "TeamID", "OppID"]

opp_for_losers = regular[["Season", "LTeamID", "WTeamID"]].copy()
opp_for_losers.columns = ["Season", "TeamID", "OppID"]

opp_games = opp_games.merge(base_winpct, on=["Season", "OppID"], how="left")
sos = opp_games.groupby(["Season", "OppID"])["OppWinPct"].mean().reset_index()
opp_winpct_lookup = base_winpct.copy().rename(columns={"TeamID": "OppID", "WinPct": "OppWinPct"})

all_opps = pd.concat([opp_for_winners, opp_for_losers], ignore_index=True)

all_opps = all_opps.merge(
    opp_winpct_lookup,
    on=["Season", "OppID"],
    how="left"
)

sos = (
    all_opps.groupby(["Season", "TeamID"])["OppWinPct"]
    .mean()
    .reset_index()
    .rename(columns={"OppWinPct": "StrengthOfSchedule"})
)

print("sos columns:", sos.columns.tolist())  # should now show TeamID

team_stats = team_stats.merge(sos, on=["Season", "TeamID"], how="left")
team_stats["StrengthOfSchedule"] = team_stats["StrengthOfSchedule"].fillna(0.5)

team_stats["EloRating"] = team_stats["EloRating"].fillna(1500)
# ELO end


train_a = build_matchup_rows(tourney, True)
train_b = build_matchup_rows(tourney, False)
train_df = pd.concat([train_a, train_b], ignore_index=True)

team1_stats = team_stats.copy().add_prefix("T1_")
team2_stats = team_stats.copy().add_prefix("T2_")

train_df = train_df.merge(
    team1_stats,
    left_on=["Season", "Team1"],
    right_on=["T1_Season", "T1_TeamID"],
    how="left"
)

train_df = train_df.merge(
    team2_stats,
    left_on=["Season", "Team2"],
    right_on=["T2_Season", "T2_TeamID"],
    how="left"
)


feature_pairs = [
    ("SeedNum", "SeedDiff"),
    ("WinPct", "WinPctDiff"),
    ("AvgPointsFor", "PointsForDiff"),
    ("AvgPointsAgainst", "PointsAgainstDiff"),
    ("AvgScoreMargin", "ScoreMarginDiff"),
    ("EloRating", "EloDiff"),
    ("StrengthOfSchedule", "SoSDiff"),
    ("LateWinPct", "LateWinPctDiff")
]

for base_col, diff_col in feature_pairs:
    train_df[diff_col] = train_df[f"T1_{base_col}"] - train_df[f"T2_{base_col}"]

feature_cols = [diff_name for _, diff_name in feature_pairs]

train_df = train_df.dropna(subset=feature_cols + ["Label"])

print(train_df)
X = train_df[feature_cols]
y = train_df["Label"]

last_season = train_df["Season"].max()

train_mask = train_df["Season"] < last_season
valid_mask = train_df["Season"] == last_season

X_train = X[train_mask]
y_train = y[train_mask]
X_valid = X[valid_mask]
y_valid = y[valid_mask]

model = LogisticRegression(max_iter=1000)
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

valid_preds = model.predict_proba(X_valid)[:, 1]
print("Validation Log Loss:", log_loss(y_valid, valid_preds))

submission = sample.copy()

id_parts = submission["ID"].str.split("_", expand=True)
submission["Season"] = id_parts[0].astype(int)
submission["Team1"] = id_parts[1].astype(int)
submission["Team2"] = id_parts[2].astype(int)

submission = submission.merge(
    team1_stats,
    left_on=["Season", "Team1"],
    right_on=["T1_Season", "T1_TeamID"],
    how="left"
)

submission = submission.merge(
    team2_stats,
    left_on=["Season", "Team2"],
    right_on=["T2_Season", "T2_TeamID"],
    how="left"
)

for base_col, diff_col in feature_pairs:
    submission[diff_col] = submission[f"T1_{base_col}"] - submission[f"T2_{base_col}"]

# Fill any missing values conservatively
for col in feature_cols:
    submission[col] = submission[col].fillna(0)

submission["Pred"] = model.predict_proba(submission[feature_cols])[:, 1]

out = submission[["ID", "Pred"]].copy()

team_names_1 = teams.rename(columns={
    "TeamID": "Team1",
    "TeamName": "Team1Name"
})

team_names_2 = teams.rename(columns={
    "TeamID": "Team2",
    "TeamName": "Team2Name"
})

submission = submission.merge(team_names_1, on="Team1", how="left")
submission = submission.merge(team_names_2, on="Team2", how="left")

Path(base_path + "/submissions").mkdir(parents=True, exist_ok=True)

out.to_csv(base_path + "/submissions/submission.csv", index=False)

# With team names
debug = submission[[
    "Season",
    "Team1Name",
    "Team2Name",
    "Pred"
]]

debug.to_json(base_path + "/submissions/debug_predictions.json", orient="records", indent=2)
# End team name


print("Saved submission.csv")
print(out.head())
