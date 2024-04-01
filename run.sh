

for ((epoch=1000; epoch<=2000; epoch+=50)); do
    file="result-${epoch}.txt"
    echo "run epoch: ${epoch}" > ${file}
    python3 -m src.test --config config/oem.yaml --use_my_query_label --opts adapt_iter ${epoch} | tee -a ${file}
done

optuna自动调参：
pip install optuna
python -m src.test_tune --config config/oem_best_newquery.yaml --use_my_query_label --opts gpus \[0\]

正常运行带测试：
python -m src.test --config config/oem_best_newquery.yaml --use_my_query_label --opts gpus \[0\]

标记图片不带测试：
python -m src.test --config config/oem_newquery.yaml --use_my_query_label --opts gpus \[0\]












