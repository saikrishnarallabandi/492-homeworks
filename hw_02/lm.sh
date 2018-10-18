



## LM
# Create an empty directory to store the whole text corpus
mkdir -p data/local/tmp
cut -f2- -d' ' < data/text > data/local/tmp/corpus.txt

# First build an LM using SRILM and store it in arpa format.
ngram-count  -order 3 -write-vocab data/local/vocab-full.txt -wbdiscount -interpolate -text data/local/tmp/corpus.txt -lm data/local/lm.arpa

# Dictionary We need a pronunciation dictionary. Lets use CMU dictionary for now 
mkdir -p data/local/dict
wget http://tts.speech.cs.cmu.edu/rsk/asr_stuff/ASRChallenge_Microsoft/resources_provided/telugu_lexicon_CMU_Indic_frontend_final.txt
cp telugu_dictionary_IITM_CommonLabelSet_final.txt data/local/dict/tel.dict

# We need to detect and specify out of vocabulary words
awk 'NR==FNR{words[$1]; next;} !($1 in words)' data/local/dict/tel.dict data/local/vocab-full.txt | egrep -v '<.?s>' > data/local/dict/vocab-oov.txt
awk 'NR==FNR{words[$1]; next;} ($1 in words)' data/local/vocab-full.txt data/local/dict/tel.dict | egrep -v '<.?s>' > data/local/dict/lexicon-iv.txt

# Lets create a lexicon
cat data/local/dict/vocab-oov.txt data/local/dict/lexicon-iv.txt | sort > data/local/dict/lexicon.txt
( echo SIL; echo SPN ) > data/local/dict/silence_phones.txt
echo SIL > data/local/dict/optional_silence.txt

# Declare non silense phones
grep -v -w sil data/local/dict/lexicon.txt | awk '{for(n=2;n<=NF;n++) { p[$n]=1; }} END{for(x in p) {print x}}'  | sort > data/local/dict/nonsilence_phones.txt

echo "--- Adding SIL to the lexicon ..."
echo -e "!SIL\tSIL" >> data/local/dict/lexicon.txt

cp data/local/dict/lexicon.txt data/local/dict/lexicon.txt.xxx
cat data/local/dict/lexicon.txt.xxx | sed 's/-pau-/-pau- SIL/' | sed 's/<unk>/<unk> SPN/' > data/local/dict/lexicon.txt

./utils/prepare_lang.sh data/local/dict '<unk>' data/local/lang data/lang || exit 0
touch  data/local/dict/extra_questions.txt

# FST
test=data/lang_test
mkdir -p $test
for f in phones.txt words.txt phones.txt L.fst L_disambig.fst phones; do     cp -r data/lang/$f $test; done
cat data/local/lm.arpa | arpa2fst --disambig-symbol=#0 --read-symbol-table=data/lang_test/words.txt - data/lang_test/G.fst
fstisstochastic data/lang_test/G.fst 

