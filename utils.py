from subprocess import check_output


def run_yolo(img_path,
             threshold,
             dataset_config_file,
             model_config_file,
             model_weights
             ):
    check_output(
        f"./darknet detector test {dataset_config_file} {model_config_file} "
        f"{model_weights} {img_path} -dont_show -thresh {threshold}",
        shell=True)
