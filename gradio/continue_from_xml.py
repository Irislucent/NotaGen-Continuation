import argparse
import os
import sys
import re
from fractions import Fraction

from abc_xml_converter import convert_abc2xml
from abc_xml_converter import convert_xml2abc


def find_voice_positions(abc: str, n: int):
    """
    returns a list of [(is_declaration, start_index, end_index)]
    """
    results = []
    for m in re.finditer(rf"(?m)^V:{n}\b[^\n]*", abc):
        line = m.group(0)
        start, end = m.span()
        after = line[len(f"V:{n}") :].strip()
        is_decl = bool(after)
        k_pos = abc.find("K:")
        if k_pos != -1 and start < k_pos:
            is_decl = True
        results.append((is_decl, start, end))
    return results


def extract_voice(abc: str, n: int) -> str:
    """
    extract the content of voice n from the ABC string.
    """
    voice_lines = find_voice_positions(abc, n)
    if not voice_lines:
        return ""

    # find the start of the first content line
    content_starts = [end for (is_decl, start, end) in voice_lines if not is_decl]
    if not content_starts:
        return ""

    start = content_starts[0]
    # find the start of the next voice declaration
    m_next = re.search(r"(?m)^V:\d+\b", abc[start:])
    end = start + m_next.start() if m_next else len(abc)
    return abc[start:end].strip()


def normalize_L_segments(abc_text: str, target_L: str = "1/8") -> str:
    """
    Unify all local L: fields into a single global target_L value,
    rescaling note durations proportionally to keep rhythmic timing consistent.
    """

    def parse_L(val: str) -> Fraction:
        num, den = val.split("/")
        return Fraction(int(num), int(den))

    target = parse_L(target_L)

    # Locate all L: fields
    segments = [
        (m.start(), parse_L(m.group(1)))
        for m in re.finditer(r"^L:([\d/]+)", abc_text, flags=re.MULTILINE)
    ]
    if not segments:
        return abc_text

    # Sentinel for end of text
    segments.append((len(abc_text), target))

    unified_parts = []
    for i in range(len(segments) - 1):
        start = segments[i][0]
        end = segments[i + 1][0]
        local_L = segments[i][1]
        scale = local_L / target
        block = abc_text[start:end]

        # Remove L: line
        block = re.sub(r"^L:.*$", "", block, flags=re.MULTILINE)

        # Scale numeric duration tokens
        def scale_len(match):
            val = match.group(1)
            val_frac = Fraction(val) if val else Fraction(1)
            new_val = val_frac * scale
            return (
                ""
                if new_val == 1
                else str(float(new_val) if new_val.denominator != 1 else int(new_val))
            )

        block = re.sub(r"(\d+(/\d+)?)", scale_len, block)
        unified_parts.append(block)

    unified_text = "\nL:%s\n" % target_L + "".join(unified_parts)
    return unified_text


def adjust_tempo_for_L(tempo, old_L, new_L):
    """If the Q field uses old_L as its base, replace with new_L."""
    if tempo.startswith(f"{old_L}="):
        return tempo.replace(f"{old_L}=", f"{new_L}=")
    return tempo


def abc_formatting(abc_str):
    """
    Normalize an ABC string to an absolute 'format 1 shell'-compatible structure.
    Ensures:
      - L:1/8 is enforced
      - Removes visual/engraving directives
      - Keeps tempo, meter, and key
      - Recomputes placeholder bar length (xN) based on meter
    """
    # Step 0: Normalize L: segments
    # abc = normalize_L_segments(abc_str) # solved by adding a parameter in convert_xml2abc

    # Step 1: Clean
    abc = re.sub(r"^\s*(I:.*|U:.*)\s*$", "", abc_str, flags=re.MULTILINE)
    abc = re.sub(r'"_.*?"', "", abc)  # Remove lyrics or underscores
    abc = abc.replace("$", "")  # Remove line break controls
    abc = re.sub(r"%.*$", "", abc, flags=re.MULTILINE)
    abc = re.sub(r"[ \t]+$", "", abc, flags=re.MULTILINE)
    abc = re.sub(r"^\s*\n", "", abc, flags=re.MULTILINE)

    # Step 2: Read M: (meter) and L: (default note length)
    def find_field(key, default=None):
        m = re.search(rf"^{key}:(.*)$", abc, flags=re.MULTILINE)
        return m.group(1).strip() if m else default

    meter = find_field("M", "4/4")
    key = find_field("K", "C")
    tempo = find_field("Q", "1/4=120")
    base_l = find_field("L", "1/8")

    tempo = adjust_tempo_for_L(tempo, base_l, "1/8")

    # Step 3: Compute the number of eighth notes per measure
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", meter)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        n8 = int(round(num * (8 / den)))
    else:
        n8 = 8
    placeholder = f"x{n8}"

    # Step 4: Normalize L: to 1/8
    if "L:" in abc:
        abc = re.sub(r"^L:.*$", "L:1/8", abc, flags=re.MULTILINE)
    else:
        # If no L: is present, add it before Q: or M:, or at the top
        if "Q:" in abc:
            abc = re.sub(r"^(Q:.*)$", r"L:1/8\n\1", abc, flags=re.MULTILINE)
        elif "M:" in abc:
            abc = re.sub(r"^(M:.*)$", r"L:1/8\n\1", abc, flags=re.MULTILINE)
        else:
            abc = "L:1/8\n" + abc

    v1 = extract_voice(abc, 1)
    v2 = extract_voice(abc, 2)

    if not v1 and not v2:
        body = re.sub(r"^[A-Z]:.*$", "", abc, flags=re.MULTILINE).strip()
        v1 = body

    # Step 6: break into bars
    def bars(s):
        bars_ = [b.strip() for b in re.split(r"\|+", s)]
        # remove trailing empty or bracket-only fragments
        while bars_ and (not bars_[-1] or bars_[-1] in ("]", "]|", "|]")):
            bars_.pop()
        return bars_

    bars_v1 = bars(v1)
    bars_v2 = bars(v2)
    max_len = max(len(bars_v1), len(bars_v2), 1)

    def get_bar(bars, i):
        return bars[i] if i < len(bars) else placeholder

    # Step 7: generate multivoice lines
    lines = []
    for i in range(max_len):
        v1_bar = get_bar(bars_v1, i)
        v2_bar = get_bar(bars_v2, i)
        line = (
            f"[V:1]{v1_bar}|[V:2]{v2_bar}|"
            f"[V:3]{placeholder}|[V:4]{placeholder}|"
            f"[V:5]{placeholder}|[V:6]{placeholder}|[V:7]{placeholder}|"
        )
        lines.append(line)

    # Step 8: organize header
    header = (
        "X:1\n"
        "%%score { ( 1 3 5 6 ) | ( 2 4 7 ) }\n"
        "L:1/8\n"
        f"Q:{tempo}\n"
        f"M:{meter}\n"
        f"K:{key}\n"
        'V:1 treble nm="Piano" snm="Pno."\n'
        "V:3 treble\n"
        "V:5 treble\n"
        "V:6 treble\n"
        "V:2 bass\n"
        "V:4 bass\n"
        "V:7 bass\n"
    )

    return header + "\n".join(lines) + "\n"


def convert_abc_str_to_xml_file(abc_str, output_directory="./"):
    convert_abc2xml(
        file_to_convert=abc_str,
        output_directory=output_directory,
        file_to_convert_is_txt=True,
    )


def safe_convert_abc2xml(**kwargs):
    "a safe wrapper to avoid argparse conflict"
    from contextlib import contextmanager

    @contextmanager
    def silence_argv():
        original_argv = sys.argv
        sys.argv = [original_argv[0]]
        try:
            yield
        finally:
            sys.argv = original_argv

    with silence_argv():
        return convert_abc2xml(**kwargs)


def run_interface(input_xml, n_bars_continue, period, composer, instrumentation):
    abc_content = convert_xml2abc(file_to_convert=input_xml, note_length=8)
    abc_content_formatted = abc_formatting(abc_content)

    from continuation import inference_patch, read_prompt_abc_str

    result = inference_patch(
        period,
        composer,
        instrumentation,
        read_prompt_abc_str(
            abc_content_formatted,
            n_bars_continue=n_bars_continue,
        ),
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_file", type=str, required=True, help="Path to the input XML file"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        default="./",
        help="Directory to save the output ABC file",
    )
    parser.add_argument(
        "-n",
        "--n_bars_continue",
        type=int,
        default=20,
        required=False,
        help="Number of bars to continue from the input ABC file",
    )
    parser.add_argument(
        "-p",
        "--period",
        type=str,
        default="Romantic",
        required=False,
        help="Musical period for continuation",
    )
    parser.add_argument(
        "-c",
        "--composer",
        type=str,
        default="Ravel, Maurice",
        required=False,
        help="Composer style for continuation",
    )
    parser.add_argument(
        "-m",
        "--instrumentation",
        type=str,
        default="Keyboard",
        required=False,
        help="Instrumentation for continuation",
    )

    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    n_bars_continue = args.n_bars_continue
    period = args.period
    composer = args.composer
    instrumentation = args.instrumentation
    if instrumentation != "Keyboard":
        instrumentation = "Keyboard"
        print(
            f"Currently only 'Keyboard' instrumentation is supported. Overriding to 'Keyboard'."
        )

    os.makedirs(output_dir, exist_ok=True)

    abc_content = convert_xml2abc(file_to_convert=input_file, note_length=8)
    abc_content_formatted = abc_formatting(abc_content)

    from continuation import inference_patch, save_and_convert, read_prompt_abc_str

    result = inference_patch(
        period,
        composer,
        instrumentation,
        read_prompt_abc_str(
            abc_content_formatted,
            n_bars_continue=args.n_bars_continue,
        ),
    )
    save_and_convert(result, period, composer, instrumentation)
