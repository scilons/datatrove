from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader, WarcReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter

from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from stopwords import get_stopwords


def get_dataset_langs(dataset_name: str, hf_token: str) -> list:
    """
    Function that gets a list of all available configs (i.e. langs) in the tmp4b dataset in order to 
    then download its own stopwords for the Gopher Quality Filtering.
    """
    try:
        # Attempt to load the dataset without a config to get the available configs
        load_dataset(dataset_name, streaming=True, token=hf_token)
    except ValueError as e:
        # Parse the available configs from the error message
        error_message = str(e)
        if "Please pick one among the available configs:" in error_message:
            configs_start = error_message.find("[") + 1
            configs_end = error_message.find("]")
            configs = error_message[configs_start:configs_end].split(", ")
            configs = [config.strip("'") for config in configs]
            return configs
        else:
            raise e

def get_stopwords(lang_list: list) -> dict:
    """
    Takes list of available langs (i.e. configs) and returns a dict with lang codes as keys and lists of stopwords as values. 
    If no stopword list is found the list will be empty.
    """

    nltk.download('stopwords')

    stopwords_list = {lang: [] for lang in lang_list}
    
    for lang in available_langs:
        try:
            stopwords_list[lang] = get_stopwords(lang)
            except:
                if lang == 'zh-tw' or lang == 'zh-cn':
                    stopwords_list[lang] = get_stopwords('zh')
                elif lang == 'en-us':
                    stopwords_list[lang] = get_stopwords('en')
                # If available get from NTLK
                elif lang == 'bn':
                    stopwords_list[lang] = set(stopwords.words('bengali'))
                elif lang == 'et':
                    stopwords_list[lang] = set(stopwords.words('greek'))
                elif lang == 'he':
                    stopwords_list[lang] = set(stopwords.words('hebrew'))
                elif lang == 'ne':
                    stopwords_list[lang] = set(stopwords.words('nepali'))
                else:
                    pass

    return stopwords_list


def main(hf_token:str):

    # Get all configs/langs in the tmp4b datasets
    available_langs = get_dataset_langs("malteos/tmp4b", hf_token)
    stopwords_list = get_stopwords(available_langs)
    # Need to think about what happens when lang = unknown in terms of LanguageFilter and GopherQualityFilter.

    # iterate over all configs in tmp4b and apply filters
    for lang in available_langs:
        
        MAIN_OUTPUT_PATH = "" #?
        FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"
        
        main_processing_executor = SlurmPipelineExecutor(
            job_name=f"cc_{lang}",
            pipeline=[
                LanguageFilter(
                    languages=lang, # need to make sure lang codes in tmp4b and in fasttext are same
                    exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/2_non_english/",
                    output_filename="${lang}/" + "/${rank}.jsonl.gz",)
                    # folder structure: language/dump/file
                    ),
                GopherRepetitionFilter(
                    exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/3_gopher_rep/{lang}")
                    ),
                GopherQualityFilter(
                    language=lang, 
                    stop_words=stopwords_list[lang],
                    exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{lang}")
                    ),
                C4QualityFilter(
                    filter_no_terminal_punct=False,
                    exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/5_c4/{lang}"),
                    ),
                FineWebQualityFilter(
                    exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{lang}")
                    ),
                JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{lang}"),
                ],
            tasks=8000,
            time="10:00:00",
            logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{lang}",
            slurm_logs_folder=f"logs/base_processing/{lang}/slurm_logs",  # must be local
            randomize_start_duration=180,  # don't hit the bucket all at once with the list requests
            mem_per_cpu_gb=2,
            partition="batch",
            )
        
        main_processing_executor.run()

        # Minhash dedup process 
        # you can also change ngrams or the number of buckets and their size here
        minhash_config = MinhashConfig(
            use_64bit_hashes=True,  # better precision -> fewer false positives (collisions)
            num_buckets=14,
            hashes_per_bucket=8,
            n_grams=5,
            )
            
        S3_MINHASH_BASE_PATH = f"{MAIN_OUTPUT_PATH}/minhash"
        S3_LOGS_FOLDER = f"{MAIN_OUTPUT_PATH}/logs/minhash"
        LOCAL_LOGS_FOLDER = "logs/minhash"
        
        TOTAL_TASKS = 1000
        
        # this is the original data that we want to deduplicate
        INPUT_READER = JsonlReader(f"{FILTERING_OUTPUT_PATH}/output/{lang}")  # this is the output from the first part
        
        # stage 1 computes minhash signatures for each task (each task gets a set of files)
        stage1 = SlurmPipelineExecutor(
            job_name=f"mh1_{lang}",
            pipeline=[
                INPUT_READER,
                MinhashDedupSignature(
                    output_folder=f"{S3_MINHASH_BASE_PATH}/{lang}/signatures", config=minhash_config
                    ),],
            tasks=TOTAL_TASKS,
            time="5:00:00",
            partition="batch",
            logging_dir=f"{S3_LOGS_FOLDER}/signatures",
            slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/slurm_logs",
            randomize_start_duration=180,
            depends=main_processing_executor,  # only start after the first one completes
            )
            
        stage2 = SlurmPipelineExecutor(
            job_name=f"mh2_{lang}",
            pipeline=[
                MinhashDedupBuckets(
                    input_folder=f"{S3_MINHASH_BASE_PATH}/{lang}/signatures",
                    output_folder=f"{S3_MINHASH_BASE_PATH}/{lang}/buckets",
                    config=MinhashConfig(use_64bit_hashes=True),
                    ),],
            tasks=minhash_config.num_buckets * 50,  # the code supports parallelizing each bucket. here we run 50
            # workers per bucket
            randomize_start_duration=180,
            logging_dir=f"{S3_LOGS_FOLDER}/buckets",
            partition="batch",
            time="02:00:00",
            mem_per_cpu_gb=4,
            cpus_per_task=3,  # you can add run more (smaller) tasks if you do not have a lot of memory
            depends=stage1,
            )
            
        stage3 = SlurmPipelineExecutor(
            job_name=f"mh3_{lang}",
            pipeline=[
                MinhashDedupCluster(
                    input_folder=f"{S3_MINHASH_BASE_PATH}/{lang}/buckets",
                    output_folder=f"{S3_MINHASH_BASE_PATH}/{lang}/remove_ids",
                    config=minhash_config,
                    ),],
            tasks=1,  # this step runs on a single task
            logging_dir=f"{S3_LOGS_FOLDER}/clustering",
            partition="batch",
            time="30:00:00",  # and can also be quite slow. Usually not this slow though
            mem_per_cpu_gb=25,
            cpus_per_task=8,  # if you dedup a full dump, you do need a lot of memory for this one
            depends=stage2,)
            
        stage4 = SlurmPipelineExecutor(
            job_name=f"mh4_{lang}",
            pipeline=[
                INPUT_READER,
                TokensCounter(),  # you can remove this one, it's just a nice way to know how many tokens we have
                # before and after dedup
                MinhashDedupFilter(input_folder=f"{S3_MINHASH_BASE_PATH}/{lang}/remove_ids"),
                # run the PII removal
                PIIFormatter(),
                JsonlWriter(f"{S3_MINHASH_BASE_PATH}/{lang}/deduped_output"),
                ],
            tasks=TOTAL_TASKS,
            logging_dir=f"{S3_LOGS_FOLDER}/filtering",
            partition="batch",
            time="5:00:00",
            mem_per_cpu_gb=4,
            depends=stage3,
            )
        # launch dedup pipelines
        stage4.run()


# Classify 'unknown'?






if __name__ == '__main__':
    # Add smth that receive the token as arg
    main(token)