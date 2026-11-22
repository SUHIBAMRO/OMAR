"""Typeset the report's display equations with LaTeX (PNG, 300 dpi), because the
document's PDF export prints LaTeX code blocks as source text.

    python -m omar_pfem.verification_agent.report_equations

Each equation is centred on a white canvas 2000 px wide (6.67 in at 300 dpi), so it
keeps about the size of the body text when the document stretches an image to the text width.
"""
import os
import subprocess
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "advisor_feedback", "report_build", "figures",
                   "verification")
EQS = {
    "eq_gci": r"\mathrm{GCI}_{\mathrm{fine}} = \frac{F_s\,\lvert v_1 - v_2\rvert}{r_{21}^{\,p} - 1}",
    "eq_fit": r"v(h) = a + b\,h^{p}, \qquad \sigma = \sqrt{\mathrm{SSR}/(n-3)}",
    "eq_ufine": r"U_{\mathrm{fine}} = 1.25\,\lvert v_{\mathrm{fine}} - a\rvert + \sigma"
                r" + \lvert r_{\mathrm{fine}}\rvert",
}
WIDTH = 2000


def main():
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, tex in EQS.items():
            src = os.path.join(tmp, name + ".tex")
            open(src, "w").write(r"\documentclass[12pt,border=4pt]{standalone}"
                                 r"\usepackage{amsmath}\usepackage{lmodern}\begin{document}"
                                 r"$\displaystyle " + tex + r"$\end{document}")
            subprocess.run(["pdflatex", "-interaction=nonstopmode", name + ".tex"], cwd=tmp,
                           check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["pdftoppm", "-png", "-r", "370", "-singlefile", name + ".pdf",
                            os.path.join(tmp, name)], cwd=tmp, check=True)
            eq = Image.open(os.path.join(tmp, name + ".png")).convert("RGB")
            canvas = Image.new("RGB", (WIDTH, eq.height + 40), "white")
            canvas.paste(eq, ((WIDTH - eq.width) // 2, 20))
            canvas.save(os.path.join(OUT, name + ".png"), dpi=(300, 300))
            print("wrote", name + ".png")


if __name__ == "__main__":
    main()
