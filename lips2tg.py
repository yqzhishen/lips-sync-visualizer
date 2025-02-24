import pandas as pd
import textgrid


def read_csv_file(csv_file):
    df = pd.read_csv(csv_file)
    time = df['TimeStamp'].values
    jaw_open = df['jawOpen'].values
    mouth_close = df['mouthClose'].values
    return time, jaw_open, mouth_close


def create_textgrid(csv_file, audio_file, output_file):
    time, jaw_open, mouth_close = read_csv_file(csv_file)

    tg = textgrid.TextGrid()
    jaw_open_tier = textgrid.PointTier(name="Jaw Open", minTime=0, maxTime=time[-1])
    mouth_close_tier = textgrid.PointTier(name="Mouth Close", minTime=0, maxTime=time[-1])

    for t, jaw in zip(time, jaw_open):
        jaw_open_tier.add(t, str(jaw))

    for t, mouth in zip(time, mouth_close):
        mouth_close_tier.add(t, str(mouth))

    tg.append(jaw_open_tier)
    tg.append(mouth_close_tier)

    tg.write(output_file)


if __name__ == "__main__":
    csv_file = r"C:\Users\yqzhishen\Desktop\2025-02-05_00-41-50\mouth_data.csv"
    audio_file = r"C:\Users\yqzhishen\Desktop\2025-02-05_00-41-50\audio.wav"
    output_file = r"C:\Users\yqzhishen\Desktop\2025-02-05_00-41-50\audio.TextGrid"
    create_textgrid(csv_file, audio_file, output_file)
