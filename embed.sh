export KMER=6
export MODEL_PATH=input/dnabert_model/
export DATA_FILE=input/atac_sequences.txt
export OUTPUT_PATH=atac_dna_embeddings.torch

python generate_embeddings.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --data_file=$DATA_FILE \
    --model_name_or_path $MODEL_PATH \
    --per_gpu_batch_size=512   \
    --n_process 24 \
    --output_file $OUTPUT_PATH \
    #--max_seq_length 158 \
