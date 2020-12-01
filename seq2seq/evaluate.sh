RESULTS=$1  # without the .txt
NAME=$2 #model name

INFILE=$RESULTS.txt
OUTFILE=$RESULTS.out.txt
SUMMARYFILE=$RESULTS.out.summary.txt
echo $INFILE
echo $OUTFILE
echo $SUMMARYFILE

python print_output.py $INFILE $OUTFILE # will write to outfile
python score.py $NAME $OUTFILE $SUMMARYFILE
