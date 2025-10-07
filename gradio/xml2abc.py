from abc_xml_converter import convert_abc2xml
from abc_xml_converter import convert_xml2abc

# Prepare ABC and XML files as strings


def read_abc_file(filename, errmsg="read error: "):
    try:
        if filename == "-.abc":
            import sys

            content = sys.stdin.read()
        else:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
        return content
    except Exception as e:
        print(errmsg + repr(e) + " " + filename)
        return None


abc_file_txt = read_abc_file("../JD_1.abc", errmsg="read error: ")


def read_xml_file(filename, errmsg="read error: "):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(errmsg + repr(e) + " " + filename)
        return None


# xml_file_txt = read_xml_file("files/xml_files/Bach 2.xml", errmsg="read error: ")


def main():
    # # 1) Example usage of convert_abc2xml

    # # a) Converting ABC to XML and returning as string

    # print("Converting ABC to XML and returning as string:")

    # xml_content = convert_abc2xml(file_to_convert="files/abc/Bach 1.abc")
    # print(xml_content)

    # # b) Converting ABC to XML and saving to directory

    # convert_abc2xml(
    #     file_to_convert="yo, i dont know_cleaned.abc", output_directory="./"
    # )

    # # c) Converting ABC string to XML and returning as string

    # print("Converting ABC string to XML and returning as string:")

    # xml_content = convert_abc2xml(
    #     file_to_convert=abc_file_txt, file_to_convert_is_txt=True
    # )
    # print(xml_content)

    # # d) Converting ABC string to XML and saving to directory:")

    convert_abc2xml(
        file_to_convert=abc_file_txt,
        output_directory="./",
        file_to_convert_is_txt=True,
    )

    # # 2) Example usage of convert_xml2abc

    # # a) Converting XML to ABC and returning as string

    # print("\nConverting XML to ABC and returning as string:\n")

    # abc_content = convert_xml2abc(file_to_convert="files/xml_files/Bach 2.xml")
    # print(abc_content)

    # # b) Converting XML to ABC and saving to directory

    # convert_xml2abc(
    #     file_to_convert="../JD_1.xml",
    #     output_directory="./",
    # )

    # # c) Converting XML string to ABC and returning as string

    # print("\nConverting XML string to ABC and returning as string:\n")

    # abc_content = convert_xml2abc(
    #     file_to_convert=xml_file_txt, file_to_convert_is_txt=True
    # )
    # print(abc_content)

    # # d) Converting XML string to ABC and saving to directory

    # convert_xml2abc(
    #     file_to_convert=xml_file_txt,
    #     output_directory="files/abc_converted",
    #     file_to_convert_is_txt=True,
    # )


if __name__ == "__main__":
    main()
