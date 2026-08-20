from dataclasses import dataclass, field

@dataclass
class Config: # single source of truth

    # data
    dataset_path: str = "/home/simon/Documents/Machine learning Seminar/Datasets/survey_vapi_messages_linked.csv"
    sb10k_train: str = "/home/simon/Documents/Machine learning Seminar/Datasets/train.tsv"
    sb10k_test: str = "/home/simon/Documents/Machine learning Seminar/Datasets/test.tsv"
    output_path: str = "/home/simon/Documents/Machine learning Seminar/annotation_data.csv"
    annotations_path: str = "/home/simon/Documents/Machine learning Seminar/annotation_data_annotated.csv"

    # deterministic directories
    output_directory_logistic_deterministic: str = "/home/simon/Documents/Machine learning Seminar/Models/Logistic Regression Models Deterministic"
    output_directory_encoder_deterministic: str = "/home/simon/Documents/Machine learning Seminar/Models/BERT Transformer Models Deterministic"
    output_directory_foundation_deterministic: str = "/home/simon/Documents/Machine learning Seminar/Models/Foundation Models Deterministic"

    # non-deterministic directories
    output_directory_logistic_nondeterministic: str = "/home/simon/Documents/Machine learning Seminar/Models/Logistic Regression Models Non-Deterministic"
    output_directory_encoder_nondeterministic: str = "/home/simon/Documents/Machine learning Seminar/Models/BERT Transformer Models Non-Deterministic"
    output_directory_foundation_nondeterministic: str = "/home/simon/Documents/Machine learning Seminar/Models/Foundation Models Non-Deterministic"

    dataset_columns: list[str] = field(default_factory = lambda: ["interviewId",
                                                                  "messageId",
                                                                  "role",
                                                                  "message"])
    seed: int = 12011853
    sample_size: int = 100  # gold-standard validation sample size

    # inference dataset
    min_words: int = 10  # minimum number of words per message

    # general model info
    n_models: int = 10 # number of checkpoints per architecture
    n_runs: int = 20 # number of inference runs per checkpoint

    # logistic regression
    name_prefix_logistic_deterministic: str = "logistic_regression_deterministic_"
    name_prefix_logistic_nondeterministic: str = "logistic_regression_nondeterministic_"
    tfidf_params: dict = field(default_factory = lambda: {"ngram_range": (1, 2), # unigrams + bigrams
                                                          "max_features": 20000,
                                                         "sublinear_tf": True, # apply log-scaling: 1 + log(tf)
                                                         "stop_words": []})

    clf_params: dict = field(default_factory = lambda: {"C": 1.0, # regularization strength
                                                       "max_iter": 1000,
                                                       "solver": "lbfgs"})

    #encoder transformer
    name_prefix_encoder_deterministic: str = "BERT-Base-German_"
    name_prefix_encoder_nondeterministic: str = "BERT-Base-German_"
    name: str = "BERT-Base-German"
    model_checkpoint: str = "google-bert/bert-base-german-cased"
    num_labels: int = 3 # number of classes
    max_length: int = 128 # max number of tokens
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    weight_decay: float = 0.01
    patience: int = 2 # training will stop early if fulfilled
    dataloader_num_workers: int = 0 # avoid non-determinism through worker reseeding
    use_bf16: bool = True # set to True for mixed precision with BF16
    use_tf32: bool = False # set to False, as True is not covered by Pytorch determinism

    #instruct-based foundational model with LoRA
    name_prefix_foundation_deterministic: str = "Qwen2-05B-Foundation_"
    name_prefix_foundation_nondeterministic: str = "Qwen2-05B-Foundation_"
    model_checkpoint_foundation: str = "Qwen/Qwen2.5-0.5B-Instruct" # small model to save compute
    max_length_foundation: int = 512
    learning_rate_foundation: float = 1e-4 #higher than BERT because of LoRA
    batch_size_foundation: int = 8
    num_epochs_foundation: int = 3
    weight_decay_foundation: float = 0.01
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    label_set_foundation: tuple = ("negativ", "neutral", "positiv") # three classes
    do_sample: bool = False # set False for greedy decoding
    use_bf16_foundation: bool = True # set to True for mixed precision with BF16
    use_tf32_foundation: bool = True # set to true for faster 32-bit matmuls
    dataloader_num_workers_foundation: int = 0



    # set near-determinism
    deterministic: bool = True

    # technical parameters
    device: str = "cuda"

# now we can import Config as cfg
cfg = Config()
