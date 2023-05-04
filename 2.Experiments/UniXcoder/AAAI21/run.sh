export CUDA_VISIBLE_DEVICES=$1

# Config
task_tag=$2
data_dir=$3
train_data=$4
valid_data=$5
test_data=$6
model_name=$7

batch_size=32
accm_steps=2
beam_size=10
source_length=512
target_length=100
output_dir=saved_models/$task_tag
train_file=$data_dir/$train_data
dev_file=$data_dir/$valid_data
test_file=$data_dir/$test_data
epochs=20 
# model_name=microsoft/unixcoder-base

# Training
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path $model_name \
	--train_filename $train_file\
	--dev_filename $dev_file \
	--output_dir $output_dir \
	--max_source_length $source_length \
	--max_target_length $target_length \
	--beam_size $beam_size \
	--train_batch_size $batch_size \
	--eval_batch_size $batch_size \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps $accm_steps \
	--num_train_epochs $epochs 

# Output results
python run.py \
	--do_test \
	--model_name_or_path $model_name \
	--test_filename $test_file \
	--output_dir $output_dir \
	--max_source_length $source_length \
	--max_target_length $target_length \
	--beam_size $beam_size \
	--train_batch_size $batch_size \
	--eval_batch_size $batch_size \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps $accm_steps \
	--num_train_epochs $epochs 