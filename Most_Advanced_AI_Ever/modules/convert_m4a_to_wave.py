# convert_m4a_to_wav.py
import os
from pydub import AudioSegment

def convert_m4a_to_wav(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.lower().endswith(".m4a"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".wav")

            try:
                audio = AudioSegment.from_file(input_path, format="m4a")
                audio.export(output_path, format="wav")
                print(f"Converted: {input_path} → {output_path}")
            except Exception as e:
                print(f"Error converting {input_path}: {e}")

if __name__ == "__main__":
    # Original dataset
    male_input = r"D:\Python_Env\New_Virtual_Env\Alfred_Offline_New_GUI\2025_09_10a_WEBUI_NEW_TRACKING_Best_so_FAR\New_V2_Home_Head_Movement_Smoothing\modules\data\males"
    female_input = r"D:\Python_Env\New_Virtual_Env\Alfred_Offline_New_GUI\2025_09_10a_WEBUI_NEW_TRACKING_Best_so_FAR\New_V2_Home_Head_Movement_Smoothing\modules\data\females"

    # Converted dataset
    male_output = r"D:\Python_Env\New_Virtual_Env\Alfred_Offline_New_GUI\2025_09_10a_WEBUI_NEW_TRACKING_Best_so_FAR\New_V2_Home_Head_Movement_Smoothing\modules\data\wave\males"
    female_output = r"D:\Python_Env\New_Virtual_Env\Alfred_Offline_New_GUI\2025_09_10a_WEBUI_NEW_TRACKING_Best_so_FAR\New_V2_Home_Head_Movement_Smoothing\modules\data\wave\females"

    convert_m4a_to_wav(male_input, male_output)
    convert_m4a_to_wav(female_input, female_output)
