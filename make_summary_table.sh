#!/bin/bash

BASE_DIR="eval_results/xluo"
OUT="$BASE_DIR/combined_table.csv"

# Hardcoded categories
categories="biology,business,chemistry,computer science,economics,engineering,health,history,law,math,other,philosophy,physics,psychology"

# Write CSV header
echo "Model,$categories,Overall" > "$OUT"

# Process each model directory
for i in 1 2 3 4; do
    summary_file=$(find "$BASE_DIR/mmlu-pro_pruned_$i/summary" -name "*summary.txt")

    model_name=$(basename "$summary_file" | sed 's/_summary.txt//')

    # Extract per-category accuracies in hardcoded order
    # Using grep + awk for each category
    row="$model_name"
    for cat in biology business chemistry "computer science" economics engineering health history law math other philosophy physics psychology; do
        val=$(grep "Average accuracy" "$summary_file" | grep "$cat" | awk '{print $3}')
        row="$row,$val"
    done

    # Extract overall accuracy
    overall=$(grep "Average accuracy:" "$summary_file" | tail -n1 | awk '{print $3}')
    row="$row,$overall"

    echo "$row" >> "$OUT"
done

echo "CSV table written to: $OUT"

