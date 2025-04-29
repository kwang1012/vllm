model_sizes=(1b 8b 32b 70b)
metrics=(throughput itl)

main() {
  for model_size in ${model_sizes[@]}; do
    for metric in ${metrics[@]}; do
      echo "Plotting $metric for $model_size"
      python experiments/plot_model.py $metric $model_size &
    done
  done
}

main "$@"