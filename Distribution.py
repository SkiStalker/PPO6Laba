import os
import shutil


def copy_files_at_percent(src_path: str, dst_path: str, bot_per: float, top_per: float):
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

    dir_list = os.listdir(src_path)
    for d in dir_list:
        cur_path = f"{src_path}/{d}"
        if os.path.isdir(cur_path):
            if not os.path.isdir(f"{dst_path}"):
                os.mkdir(dst_path)

            if not os.path.isdir(f"{dst_path}/{d}"):
                os.mkdir(f"{dst_path}/{d}")
            files = [f for f in os.listdir(cur_path) if os.path.isfile(os.path.join(cur_path, f))]
            for i, file in enumerate(files):
                if bot_per <= (i / len(files)) < top_per:
                    shutil.copyfile(os.path.join(cur_path, file), f"{dst_path}/{d}/{file}")


if __name__ == '__main__':
    path = "./data"
    train_path = f"{path}/train"
    valid_path = f"{path}/val"
    test_path = f"{path}/test"

    copy_files_at_percent(path, train_path, 0, 0.8)
    copy_files_at_percent(path, valid_path, 0.8, 0.9)
    copy_files_at_percent(path, test_path, 0.9, 1.1)
