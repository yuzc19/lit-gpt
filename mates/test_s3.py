import copy

from datasets import DownloadConfig, load_dataset


class ReviseDownloadConfig(DownloadConfig):
    def __post_init__(self, use_auth_token):
        if use_auth_token != "deprecated":
            warnings.warn(
                "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'token={use_auth_token}' instead.",
                FutureWarning,
            )
            self.token = use_auth_token

    def copy(self):
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})


downloadconfig = ReviseDownloadConfig()
load_dataset(
    "json",
    data_files=[
        "s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_1_of_10/shard_00000000_processed.jsonl.zstd"
    ],
    download_config=downloadconfig,
)
