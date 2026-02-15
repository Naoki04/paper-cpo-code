import cairosvg
import argparse
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert SVG to EPS")
    parser.add_argument("svg_dir", type=str, help="Path to dir ofthe SVG file")
    
    
    svg_dir = parser.parse_args().svg_dir
    
    # SVGファイルのパスを取得
    svg_paths = []
    for root, dirs, files in os.walk(svg_dir):
        for file in files:
            if file.endswith(".svg"):
                svg_paths.append(os.path.join(root, file))
                
    for svg_path in tqdm(svg_paths):
        # 拡張子を変え、./pdfに保存
        output_path = os.path.join(
            os.path.dirname(svg_path), os.path.basename(svg_path).replace(".svg", ".pdf")
        )

        cairosvg.svg2pdf(
            url=svg_path,
            write_to=output_path
        )

    
if __name__ == "__main__":
    main()
    


