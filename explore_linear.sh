for alpha in 0 0.2 0.5 0.7 ; do
  for l1 in 0.2 0.5 0.7 ; do
    echo "======================================"
    (set -x; python mlflow_linear.py $alpha $l1)
    echo "======================================"
    echo
  done
done
