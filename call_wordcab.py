import json
import time

from wordcab import start_summary, retrieve_job, retrieve_summary
from wordcab.core_objects import GenericSource, InMemorySource


# with open("data/lars_sample.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# format the file as a txt
# for utterance in data["utterances"]:
#     speaker = "Speaker A" if utterance["speaker"] == 0 else "Speaker B"
#     text_to_insert = f"{speaker}: {utterance['text']}\n"

#     with open("data/lars_sample.txt", "a", encoding="utf-8") as f:
#         f.write(text_to_insert)

# with open("lars.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# for utt in data["transcript"]:
#     # Replace Contact with Speaker A and Caller with Speaker B
#     utt.replace("Contact", "Speaker A")
#     utt.replace("Caller", "Speaker B")

# source = InMemorySource(data)
source = GenericSource("data/lars_sample.txt")
job = start_summary(
    source_object=source,
    display_name="Generic",
    summary_type="narrative",
    context=["keywords", "next_steps", "discussion_points", "issue", "purpose"],
    source_lang="en",
    summary_lens=5,
    only_api=False,
    api_key="333ac79784518d1bbd1b59a18530665d01a52261",
)
print(job.job_name)

while True:
	job = retrieve_job(job_name=job.job_name, api_key="333ac79784518d1bbd1b59a18530665d01a52261")
	if job.job_status == "SummaryComplete":
		break
	else:
		time.sleep(3)

# Get the summary id
summary_id = job.summary_details["summary_id"]
# Retrieve the summary
summary = retrieve_summary(summary_id=summary_id, api_key="333ac79784518d1bbd1b59a18530665d01a52261")

# Get the summary as a human-readable string
print(summary.get_formatted_summaries())
