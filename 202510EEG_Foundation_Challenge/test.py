from eegdash import EEGDashDataset

DATA_DIR = '/ssd1/zhanghongbo04/002/project/AI-competition/202510EEG_Foundation_Challenge/startkit/data_test'
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# subjects = ["012", "013", "014"]
dataset_RestingState = EEGDashDataset(
    cache_dir=DATA_DIR,
    dataset="ds005509",
    # subject=subjects,
    task="RestingState"
)
print(f"Number of recordings: {len(dataset_RestingState)}")
# dataset_ccd = EEGDashDataset(
#     task="RestingState",
#     release="R5", 
#     cache_dir=DATA_DIR,
#     mini=True,
# )