import cairosvg
import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert SVG to EPS")
    parser.add_argument("svg_path", type=str, help="Path to the SVG file")
    
    svg_path = parser.parse_args().svg_path

    # 拡張子以外はそのまま
    output_path = svg_path.rsplit(".", 1)[0]+".pdf"

    cairosvg.svg2pdf(
        url=svg_path,
        write_to=output_path
    )

    
if __name__ == "__main__":
    main()
    


