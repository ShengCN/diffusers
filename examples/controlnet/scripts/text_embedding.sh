exp_name="TextEmbedding"
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="$exp_name"

batch_size=28

accelerate launch train_genlight.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=5e-4 \
 --validation_image "./Test/human2.png" "./Test/man.png" \
 --validation_prompt "0.5 0.5" "0.3 0.3" \
 --resume_from_checkpoint='latest'\
 --train_batch_size=$batch_size\
 --num_train_epochs=10000\
 --tracker_project_name "$exp_name"\
 --ibl_embedding="Text"\
 --validation_steps=100 \
 --training_steps=100 \
#  --enable_xformers_memory_efficient_attention\

