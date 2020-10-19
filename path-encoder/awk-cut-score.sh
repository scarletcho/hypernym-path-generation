#!/bin/bash
fname=$1

echo "# H@1 score (dev; is_gold_dev)"
ig_dev_true_cnt=$(cut -f 8 < $fname | tr ' ' '\n' | grep True | wc -l)
ig_dev_false_cnt=$(cut -f 8 < $fname | tr ' ' '\n' | grep False | wc -l)
echo $(awk "BEGIN {print $ig_dev_true_cnt/($ig_dev_true_cnt+$ig_dev_false_cnt); exit}")

echo -e "\n# H@1 score (WordNet; is_gold_wn)"
ig_wn_true_cnt=$(cut -f 9 < $fname | tr ' ' '\n' | grep True | wc -l)
ig_wn_false_cnt=$(cut -f 9 < $fname | tr ' ' '\n' | grep False | wc -l)
echo -e "\n# True cnt"
echo $ig_wn_true_cnt

echo -e "\n# False cnt"
echo $ig_wn_false_cnt
echo $(awk "BEGIN {print $ig_wn_true_cnt/($ig_wn_true_cnt+$ig_wn_false_cnt); exit}")

echo -e "\n# Lex-ident ratio"
true_cnt=$(cut -f 10 < $fname | tr ' ' '\n' | grep True | wc -l)
false_cnt=$(cut -f 10 < $fname | tr ' ' '\n' | grep False | wc -l)
echo $(awk "BEGIN {print $true_cnt/($true_cnt+$false_cnt); exit}")

echo -e "\n# Wu & Palmer score"
echo $(awk '{sum+=$11; cnt++} END {print sum/cnt}' $fname)

