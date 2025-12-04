import json
import os
import lzma
import json
import bz2
import zstandard as zstd
import io
import pickle
import multiprocessing
import argparse

def extract_posts_from_lines(archive_lines, output_dir, 
                             file_name_no_extension, 
                             parent_ids, reply_ids):
    output_path = os.path.join(output_dir, file_name_no_extension)
    
    # open a plaintext writer to dump posts of interested users
    with open(output_path + '_parent.temp', 'w', encoding='utf-8') as parent_f:
        with open(output_path + '_reply.temp', 'w', encoding='utf-8') as reply_f:
            for line in archive_lines:
                # if line belongs to one of the interested parent post ids, dump it out
                parent_interested = check_line_interested(line, parent_ids)
                if parent_interested:
                    parent_f.write(line)

                # if line's parents belongs to one of the interested post ids, dump it out
                reply_interested = check_line_interested_parent(line, reply_ids)
                if reply_interested:
                    reply_f.write(line)

    # rename temp files
    os.rename(output_path + '_reply.temp', output_path + '_reply')
    os.rename(output_path + '_parent.temp', output_path + '_parent')

    print("done", file_name_no_extension)

def check_line_interested_parent(line, interested_ids):
    """checks if the line's parent is one of the interested post ids

    Args:
        line (_type_): line to check
        interested_ids (_type_): set of interested post_ids

    Returns:
        bool: whether the line does belong to one of the interested users
    """
    try:    
        data = json.loads(line)
        parent_id = data["parent_id"].split('_', 1)[1]
        
        if parent_id in interested_ids:
            return True
    except Exception as e:
        print(e)
        print(line)

    return False

def check_line_interested(line, interested_ids):
    """checks if the line belongs to one of the interested ids

    Args:
        line (_type_): line to check
        interested_ids (_type_): set of interested post_ids

    Returns:
        bool: whether the line does belong to one of the interested users
    """
    try:    
        data = json.loads(line)
        post_id = data["id"]
        
        if post_id in interested_ids:
            return True
    except Exception as e:
        print(e)
        print(line)

    return False

def extract_posts(archive_path, output_dir, parent_ids, reply_ids):
    if '.' in archive_path:
        file_name_no_extension = archive_path.split(os.sep)[-1].split(".")[0]
        extension = archive_path.split(os.sep)[-1].split(".")[1]
    else:
        file_name_no_extension = archive_path.split(os.sep)[-1]
        extension = ''

    # open archive and set 
    if extension == "xz":
        with lzma.open(archive_path, 'rt') as xz_file:
            extract_posts_from_lines(xz_file, 
                                     output_dir, 
                                     file_name_no_extension, 
                                     parent_ids, reply_ids)
    elif extension == "bz2":
        with bz2.open(archive_path, 'rt') as bz_file:
            extract_posts_from_lines(bz_file, 
                                     output_dir, 
                                     file_name_no_extension, 
                                     parent_ids, reply_ids)
    elif extension == "zst":
        with open(archive_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            stream_reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
            extract_posts_from_lines(text_stream, 
                                     output_dir, 
                                     file_name_no_extension, 
                                     parent_ids, reply_ids)
    else:
        with open(archive_path, 'r', encoding='utf-8') as fh:
            extract_posts_from_lines(fh, 
                                     output_dir, 
                                     file_name_no_extension, 
                                     parent_ids, reply_ids)

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-y', type=str, dest='string_match', default=None, help='Optional string flag for matching specific archives in a given year for processing')
    parser.add_argument('-c', type=int, dest='num_cores', default=None, help='Optional int flag for max number of processes')
    args = parser.parse_args()

    with open("./config.json") as f:
        config = json.load(f)
        collection_start_date = config["collection_start_date"]
        collection_end_date = config["collection_end_date"]

        root_output_dir_path = config["root_output_dir_path"]

        comment_folder_path = root_output_dir_path + "/1_extract_user_posts"
        parent_reply_id_path = root_output_dir_path + "/1_extract_user_posts"
    
    # Get parent and child ids
    parent_ids = []
    reply_ids = []
    
    # parent ids contain a t1_ or t3_ before the id which must be removed for comparison. Reply IDs are without the tag
    def extract_after_underscore(strings):
        result = []
        for s in strings:
            if '_' in s:
                result.append(s.split('_', 1)[1])
            else:
                result.append(s)
        return result
    
    for filename in os.listdir(parent_reply_id_path):
        if filename.endswith('.pkl') or filename.endswith('.pickle'):
            full_path = os.path.join(parent_reply_id_path, filename)
            try:
                with open(full_path, 'rb') as f:
                    data = pickle.load(f)
                    if "post_id" in filename:
                        reply_ids.extend(data)
                    elif "parent_id" in filename:
                        parent_ids.extend(extract_after_underscore(data))
                    else:
                        print("there is a pickle file that shouldnt be there", filename)
            except Exception as e:
                print(f"Error reading {full_path}: {e}")

    parent_ids = set(parent_ids)
    reply_ids = set(reply_ids)

    if args.string_match is None:
        interested_years = list(range(collection_start_date, collection_end_date))
    else:
        interested_years = [args.string_match]

    # get archives
    archive_list = []
    for filename in os.listdir(comment_folder_path):
        if filename.endswith('.xz') or filename.endswith('.bz2')  or filename.endswith('.zst') or '.' not in filename:
            archive_list.append(filename)
    print(len(archive_list),"archives")


    # Create a multiprocessing pool
    if args.num_cores is None:
        concurrent_processes = multiprocessing.cpu_count()
    else:
        concurrent_processes = args.num_cores
    print("num_cores",concurrent_processes)
    pool = multiprocessing.Pool(processes=concurrent_processes)     
    
    # output folder for this step of processing
    output_dir = root_output_dir_path + "/2_extract_parent_reply"
    os.makedirs(output_dir, exist_ok=True)

    # Process each archive file using the multiprocessing 
    for filename in archive_list:
        # ignore archives of years we are not interested in
        interested_year = False
        for y in interested_years:
            if str(y) in filename:
                interested_year = True
                break
        if interested_year == False:
            print("file from non-interested year",filename)
            continue

        # list of finished filenames
        file_name_no_extension = filename.split(".")[0]
        finished_filenames = [
            os.path.join(output_dir, file_name_no_extension + "_parent"),
            os.path.join(output_dir, file_name_no_extension + "_reply")
            ]
        
        # only process if no temp or finished file exists
        if os.path.exists(os.path.join(output_dir, file_name_no_extension + "_parent.temp")) or os.path.exists(os.path.join(output_dir, file_name_no_extension + "_reply.temp")):
            print(filename, "temp exists")
            continue
        elif any(os.path.exists(path) for path in finished_filenames):
            print(filename, "has already processed files")
            continue
        else:
            print("processing",filename)
            pool.apply_async(extract_posts, (os.path.join(comment_folder_path,filename), output_dir, parent_ids, reply_ids))
            
    # Close the multiprocessing pool
    pool.close()
    pool.join()
    print("all done")

