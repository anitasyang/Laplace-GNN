export PYTHONPATH=..:$PYTHONPATH
planetoid_datasets=(cora citeseer pubmed)
webkb_datasets=(cornell texas wisconsin washington)
wiki_datasets=(chameleon squirrel)

# if [[ "$2" != "gcn" && "$2" != "stegcn" ]]; then
#     echo "Error: model_type must be either 'gcn' or 'stegcn'"
#     exit 1
# fi

if [[ " ${planetoid_datasets[@]} " =~ " $1 " ]]; then
    echo "Running with planetoid params on $1";
    if [ " $2 " = " stegcn " ]; then
        n_epochs=400
        lr=0.001
    elif [ "$2" = "gcn" ] || [ "$2" = "gat" ]; then
        n_epochs=200
        lr=0.01
    fi
    # if [ " $1 " = " citeseer " ]; then
    #     lr=0.01
    # fi
    python3 marglik_training.py \
        --dataset=$1 \
        --init_graph=$3 \
        --subset_of_weights=all \
        --hessian_structure=kron \
        --n_hypersteps=10 \
        --n_epochs_burnin=20 \
        --marglik_frequency=20 \
        --n_epochs=$n_epochs \
        --n_hyper_stop=350 \
        --model_type=$2 \
        --gpu_num=0 \
        --ste_thresh=0.5 \
        --weight_decay=5e-5 \
        --n_data_rand_splits=10 \
        --n_repeats=1 \
        --lora_r=64 \
        --weight_decay_adj=5e-4 \
        --dropout_p=0.5 \
        --hidden_channels=64 \
        --lr_adj=0.8 \
        --norm=none \
        --res=false \
        --optimizer=adam \
        --grad_norm=true \
        --momentum_adj=0.9 \
        --symmetric=false \
        --train_masked_update=false \
        --early_stop=false \
        --lr=$lr \
        # --sign_grad=true
        # --stop_criterion=valloss \
elif [[ " ${webkb_datasets[@]} " =~ " $1 " ]]; then
    echo "Running with webkb params on $1"
    if [ " $2 " = " stegcn " ]; then
        n_epochs=1000
        lr=0.01
    elif [ " $2 " = " gcn " ]; then
        n_epochs=200
        lr=0.001
    fi
    python3 marglik_training.py \
        --dataset=$1 \
        --init_graph=$3 \
        --subset_of_weights=all \
        --hessian_structure=kron \
        --n_hypersteps=10 \
        --n_epochs_burnin=5 \
        --n_hyper_stop=1000 \
        --marglik_frequency=20 \
        --model_type=$2 \
        --gpu_num=0 \
        --n_data_rand_splits=10 \
        --n_repeats=1 \
        --ste_thresh=0.5 \
        --lora_r=64 \
        --n_epochs=$n_epochs \
        --lr_adj=10 \
        --weight_decay_adj=5e-4 \
        --weight_decay=5e-4 \
        --dropout_p=0.5 \
        --hidden_channels=64 \
        --norm=none \
        --res=true \
        --train_masked_update=false \
        --symmetric=true \
        --optimizer=adam \
        --lr=$lr \
        --grad_norm=true \
        --momentum_adj=0.9 \
        --early_stop=false \
        # --stop_criterion=valloss \
elif [[ " ${wiki_datasets[@]} " =~ " $1 " ]]; then
    echo "Running with wiki params on $1"
    if [ " $2 " = " stegcn " ]; then
        n_epochs=400
        lr=0.001
        # n_hyper_stop=200
    elif [ " $2 " = " gcn " ]; then
        n_epochs=200
        lr=0.001
        # n_hyper_stop=400
    fi
    python3 marglik_training.py \
        --dataset=$1 \
        --init_graph=$3 \
        --subset_of_weights=all \
        --hessian_structure=kron \
        --n_hypersteps=10 \
        --n_epochs_burnin=50 \
        --n_hyper_stop=1000 \
        --marglik_frequency=10 \
        --model_type=$2 \
        --hidden_channels=32 \
        --gpu_num=0 \
        --ste_thresh=0.5 \
        --weight_decay=5e-5 \
        --weight_decay_adj=5e-4 \
        --dropout_p=0.5 \
        --lora_r=128 \
        --n_epochs=$n_epochs \
        --lr=$lr \
        --lr_adj=10 \
        --n_data_rand_splits=10 \
        --n_repeats=1 \
        --norm=none \
        --symmetric=true \
        --train_masked_update=false \
        --optimizer=adam \
        --grad_norm=true \
        --momentum_adj=0. \
        --early_stop=false \
        --res=false \
        # --stop_criterion=valloss \
elif [[ " actor " == " $1 " ]]; then
    echo "Running with actor params"
    python3 marglik_training.py \
        --dataset=actor \
        --model_type=$2 \
        --init_graph=$3 \
        --subset_of_weights=all \
        --hessian_structure=kron \
        --hidden_channels=32 \
        --lr=0.05 \
        --weight_decay=5e-5 \
        --dropout_p=0.5 \
        --n_epochs=200 \
        --n_hypersteps=10 \
        --n_epochs_burnin=100 \
        --marglik_frequency=20 \
        --n_repeats=10 \
        --stop_criterion=valloss \
        # --lr_adj=0.1 \
        # --ste_thresh=0.1 \
elif [[ " karate " == " $1 " ]]; then
    if [ " $2 " = " stegcn " ]; then
        n_epochs=200
        lr=0.05
    elif [ " $2 " = " gcn " ]; then
        n_epochs=200
        lr=0.05
    fi
    python3 marglik_training.py \
        --dataset=karate \
        --init_graph=original \
        --subset_of_weights=all \
        --hessian_structure=kron \
        --n_epochs=$n_epochs \
        --weight_decay=5e-5 \
        --n_hypersteps=10 \
        --n_epochs_burnin=50 \
        --marglik_frequency=20 \
        --model_type=$2 \
        --n_repeats=1 \
        --gpu_num=0 \
        --n_data_rand_splits=1 \
        --ste_thresh=0.5 \
        --lr=$lr \
        --dropout_p=0.5 \
        --weight_decay_adj=5e-5 \
        --hidden_channels=64 \
        --lr_adj=1 \
        --res=false \
        --norm=none \
        --symmetric=true \
        --lora_r=64
elif [ " banana " == " $1 " ] || [ " circle " == " $1 " ]; then
    echo "Running with $1 params"
    python3 marglik_training.py \
        --dataset=banana \
        --dataset=$1 \
        --init_graph=$3 \
        --subset_of_weights=all \
        --hessian_structure=kron \
        --n_hypersteps=10 \
        --n_epochs_burnin=50 \
        --n_hyper_stop=200 \
        --marglik_frequency=10 \
        --model_type=$2 \
        --hidden_channels=50 \
        --gpu_num=0 \
        --ste_thresh=0.5 \
        --weight_decay=5e-5 \
        --weight_decay_adj=5e-4 \
        --dropout_p=0.5 \
        --lora_r=64 \
        --n_epochs=400 \
        --lr=0.01 \
        --lr_adj=0.8 \
        --n_data_rand_splits=1 \
        --n_repeats=1 \
        --norm=none \
        --symmetric=false \
        --train_masked_update=false \
        --optimizer=adam \
        --grad_norm=true \
        --momentum_adj=0.9 \
        --early_stop=false \
        --res=false
else
    echo "Dataset not found"
    exit 1
fi
