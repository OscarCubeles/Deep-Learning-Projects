from pypdf import PdfMerger

pdf1 = f"DNI - Albert Cubeles II.pdf"
pdf2 = "DNI - Albert Cubeles.pdf"
output = "DNI_Albert.pdf"

merger = PdfMerger()
merger.append(pdf1)
merger.append(pdf2)

with open(output, "wb") as f:
    merger.write(f)

merger.close()

print("PDFs merged into", output)