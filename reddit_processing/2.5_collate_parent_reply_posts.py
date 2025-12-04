import json
import os
from archive_processing_utils import load_df_jsonl


if __name__ == "__main__":

    with open("./config.json") as f:
        config = json.load(f)
        collection_start_date = config["collection_start_date"]
        collection_end_date = config["collection_end_date"]
        root_output_dir_path = config["root_output_dir_path"]
        processed_parent_reply_path = root_output_dir_path + "/2_extract_parent_reply"
    
    # get processed archives
    archive_list = []
    for filename in os.listdir(processed_parent_reply_path):
        archive_list.append(filename)
    print(len(archive_list),"archives")
    
    
    # output folder for this step of processing
    output_dir = processed_parent_reply_path + "/2_extract_parent_reply_collated"
    os.makedirs(output_dir, exist_ok=True)

    for pr in ["parent", "reply"]:
        output_filepath = os.path.join(output_dir, pr)
        with open(output_filepath, "w") as outfile:
            for y in range(int(collection_start_date), int(collection_end_date)):
                year = str(y)
                print(year)
                for m in range(1,13):
                    month = str(m).zfill(2)
                    archive_name = "RC_" + year + "-" + month + "_" + pr
                    
                    in_filepath = os.path.join(processed_parent_reply_path, archive_name)
                    with open(in_filepath, "r") as infile:
                        for line in infile:
                            outfile.write(line)
                    print("done", archive_name)
                print("done", year, pr)

        # load and save into pkl files for faster loading in the future
        df = load_df_jsonl(output_filepath)
        df.to_pickle(os.path.join(output_dir, pr + ".pkl"))

