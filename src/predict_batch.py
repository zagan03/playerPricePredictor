import json
import pandas as pd
from predict import predict_player_market_value


def run_batch_predictions(input_json_path, output_csv_path):
    print(f"Reading data from: {input_json_path}...")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        players_data = json.load(f)

    results = []
    print("\nStarting predictions:\n" + "-" * 40)

    for player in players_data:
        name = player.pop('name', 'Unknown Player')

        # Get the predicted value
        estimated_value = predict_player_market_value(player)

        print(f"Player: {name}")
        print(f"   -> Position: {player.get('sub_position')} | Age: {player.get('age')} years")
        print(f"   -> Estimated Value: {estimated_value:.2f}M EUR\n")

        # Restore the name for export
        player['name'] = name
        player['predicted_value_m'] = round(estimated_value, 2)
        results.append(player)

    # Save results to a CSV file
    df_results = pd.DataFrame(results)

    # Move important columns to the front for better visibility
    cols = ['name', 'predicted_value_m', 'age', 'sub_position'] + [
        c for c in df_results.columns
        if c not in ['name', 'predicted_value_m', 'age', 'sub_position']
    ]
    df_results = df_results[cols]

    df_results.to_csv(output_csv_path, index=False)
    print("-" * 40 + f"\nProcessing complete! File saved to: {output_csv_path}")


if __name__ == "__main__":
    INPUT_PATH = "../data/test_players.json"
    OUTPUT_PATH = "../data/processed/batch_predictions_output.csv"

    run_batch_predictions(INPUT_PATH, OUTPUT_PATH)