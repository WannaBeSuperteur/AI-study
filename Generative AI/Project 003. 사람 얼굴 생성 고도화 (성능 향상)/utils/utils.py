import os


def get_img_path(dir_name, img_name):
    return os.path.join(dir_name, img_name).replace(os.sep, '/')