import json
import os
import lzma
import json
import bz2
import zstandard as zstd
import io
import pickle
import multiprocessing


def extract_posts_from_lines(archive_lines, output_dir, 
                             file_name_no_extension, 
                             interested_users):
    output_path = os.path.join(output_dir, file_name_no_extension)
    post_id_list = []
    parent_id_list = []
    
    # open a plaintext writer to dump posts of interested users
    with open(output_path + '.temp', 'w', encoding='utf-8') as f:
        for line in archive_lines:
            interested, parent_id, post_id = check_line_interested(line, interested_users)
            # if line belongs to one of the interested users, dump it out
            if interested:
                post_id_list.append(post_id)
                parent_id_list.append(parent_id)
                f.write(line)

    # dump parent and post id lists
    with open(os.path.join(output_dir, f'{file_name_no_extension}_post_id_list.pkl'), "wb") as f:
        pickle.dump(post_id_list, f)
    with open(os.path.join(output_dir, f'{file_name_no_extension}_parent_id_list.pkl'), "wb") as f:
        pickle.dump(parent_id_list, f)
        
    # rename temp file
    os.rename(output_path + '.temp', output_path)

    print("done", file_name_no_extension)


def check_line_interested(line, interested_users):
    """checks if the line belongs to one of the interested users

    Args:
        line (_type_): line to check
        interested_users (_type_): set of interested users

    Returns:
        bool: whether the line does belong to one of the interested users
        str: the parent id of the line
        str: the comment id of the line
    """
    try:    
        data = json.loads(line)
        user_name = data["author"]
        parent_id = data["parent_id"]
        post_id = data["id"]
        

        if user_name.lower() in interested_users:
            return True, parent_id, post_id
    except Exception as e:
        print(e)
        print(line)

    return False, None, None

def extract_posts(archive_path, output_dir, interested_users):
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
                                     interested_users)
    elif extension == "bz2":
        with bz2.open(archive_path, 'rt') as bz_file:
            extract_posts_from_lines(bz_file, 
                                     output_dir, 
                                     file_name_no_extension, 
                                     interested_users)
    elif extension == "zst":
        with open(archive_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            stream_reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
            extract_posts_from_lines(text_stream, 
                                     output_dir, 
                                     file_name_no_extension, 
                                     interested_users)
    else:
        with open(archive_path, 'r', encoding='utf-8') as fh:
            # lines = fh.readlines()
            extract_posts_from_lines(fh, 
                                     output_dir, 
                                     file_name_no_extension, 
                                     interested_users)

    


if __name__ == "__main__":

    with open("./config.json") as f:
        config = json.load(f)
        collection_start_date = config["collection_start_date"]
        collection_end_date = config["collection_end_date"]
        comment_folder_path = config["raw_comment_folder_path"]
        root_output_dir_path = config["root_output_dir_path"]
        interested_usernames_path = config["interested_usernames_path"]

    interested_years = list(range(collection_start_date, collection_end_date))

    # get archives
    archive_list = []
    for filename in os.listdir(comment_folder_path):
        if filename.endswith('.xz') or filename.endswith('.bz2')  or filename.endswith('.zst') or '.' not in filename:
            archive_list.append(filename)
    print(len(archive_list),"archives")


    # Create a multiprocessing pool
    concurrent_processes = multiprocessing.cpu_count()
    print("num_cores",concurrent_processes)
    pool = multiprocessing.Pool(processes=concurrent_processes)
    
    # output folder for this step of processing
    output_dir = root_output_dir_path + "/1_extract_user_posts"
    os.makedirs(output_dir, exist_ok=True)

    # get interested users (stored as list of strings)
    with open(interested_usernames_path, "rb") as f:
        interested_users = pickle.load(f)
        
    interested_users = set(interested_users)
    # make lower to ignore case
    interested_users = {s.lower() for s in interested_users}
    
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
            os.path.join(output_dir, file_name_no_extension),
            os.path.join(output_dir, f'{file_name_no_extension}_post_id_list.pkl'),
            os.path.join(output_dir, f'{file_name_no_extension}_parent_id_list.pkl')
            ]
        
        # only process if no temp or finished file exists
        if os.path.exists(os.path.join(output_dir, file_name_no_extension + ".temp")):
            print(filename, "temp exists")
            continue
        elif any(os.path.exists(path) for path in finished_filenames):
            print(filename, "has already processed files")
            continue
        else:
            print("processing",filename)
            pool.apply_async(extract_posts, (os.path.join(comment_folder_path,filename), output_dir, interested_users))
            
    # Close the multiprocessing pool
    pool.close()
    pool.join()
    print("all done")

