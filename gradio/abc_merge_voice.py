import re


def parse_voices(abc_text):
    # Split into voice blocks
    voices = {}
    current_voice = None
    lines = abc_text.split("\n")
    for line in lines:
        m = re.match(r"^V:(\d+)", line)
        if m:
            current_voice = m.group(1)
            voices[current_voice] = []
        elif current_voice:
            voices[current_voice].append(line)
    # Join lines per voice
    for v in voices:
        voices[v] = "\n".join(voices[v])
    return voices


def split_bars(voice_text):
    # Split by barlines (| and |$)
    bars = re.split(r"\|", voice_text)
    return [b.strip() for b in bars if b.strip()]


def merge_voices(voices_dict, voice_order):
    # Split each voice into bars
    bars_dict = {v: split_bars(voices_dict[v]) for v in voice_order}
    num_bars = max(len(bars_dict[v]) for v in voice_order)
    merged_lines = []
    for i in range(num_bars):
        line = ""
        for v in voice_order:
            bar = bars_dict[v][i] if i < len(bars_dict[v]) else "x8"
            line += f"[V:{v}]{bar}|"
        merged_lines.append(line)
    return merged_lines


def read_abc(abc_path):
    with open(abc_path, "r") as f:
        abc_text = f.read()
    return abc_text


def save_lines_abc(lines, output_path):
    with open(output_path, "w") as f:
        f.writelines(lines)


abc = read_abc(
    "/home/yuxuan.wu/NotaGen/gradio/generated/20250909_170912_Romantic_Debussy, Claude_Keyboard.abc"
)

lines = abc.strip().split("\n")
metadata = []
voices_start = 0
for i, line in enumerate(lines):
    if line.startswith("V:1"):
        voices_start = i
        break
    metadata.append(line)
metadata = [line + "\n" for line in metadata]
abc_voices = "\n".join(lines[voices_start:])


voice_order = ["1", "2", "3", "4", "5", "6"]

voices_dict = parse_voices(abc_voices)
merged_lines = merge_voices(voices_dict, voice_order)

all_lines = metadata + [""] + merged_lines

save_lines_abc(all_lines, "/home/yuxuan.wu/NotaGen/kevin_converted/kevin_merged.abc")
