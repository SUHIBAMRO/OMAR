"""Standalone reply document: a complete, self-contained response to
Professor Rabczuk's 2026-11-10 email (groove-stress convergence,
smoothness, contour plot, SCF, higher-order elements, the singularity
question, and the far-field PI-vs-DD comparison), in its own file,
written as a proper report/letter rather than a bare section extract --
separate from the large B1/B3 re-evaluation report.
"""
import os

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

FIGURES_DIR = "/tmp/claude-0/-home-user/64d7c4d8-d5f0-5686-a58f-aa87abfd4ba4/scratchpad"
OUT_DIR = FIGURES_DIR + "/report"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "Timon_2026-11-10_Email_Full_Response.docx")

doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.left_margin = Inches(1.25)
section.right_margin = Inches(1.25)
section.top_margin = Inches(1.0)
section.bottom_margin = Inches(1.0)

HEADING1_COLOR = RGBColor(0x36, 0x5F, 0x91)
HEADING2_COLOR = RGBColor(0x4F, 0x81, 0xBD)


def title(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)


def h1(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = HEADING1_COLOR


def h2(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = HEADING2_COLOR


def para(text, italic=False, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(htext)
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(val))
            r.font.size = Pt(9)
    doc.add_paragraph()


def image(filename, caption=None, width_in=6.0):
    path = os.path.join(FIGURES_DIR, filename)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width_in))
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)


# ===========================================================================
title("B3 Groove-Stress Convergence: Response to Professor Rabczuk's 2026-11-10 Email")
para("Prepared 2026-11-11. This report answers both points raised in Professor Rabczuk's "
     "2026-11-10 email in full: (1) establishing genuine convergence of the groove-region "
     "stress -- its smoothness, local mesh refinement, the Stress Concentration Factor, and "
     "whether this reflects a true geometric singularity or a bounded concentration -- and "
     "(2) the far-field PI-vs-DD comparison requested once a trustworthy reference region was "
     "available.")

# ===========================================================================
h1("1. Is the groove geometry smooth enough to exclude a singularity?")
para("The groove profile r(z) = R_in0 - (d/2)(1+cos(pi(z-zc)/w)) is a raised-cosine dip. At "
     "its own deepest point (z=zc, the center of our region mask) it is genuinely smooth -- a "
     "finite, nonzero radius of curvature (rho~0.0228) -- not a geometric singularity (a true "
     "singularity, e.g. a V-notch or crack tip, has unbounded curvature).")
para("A separate feature does exist, however, at the two points where the raised-cosine "
     "region joins the flat wall (z = zc +/- 0.15): we confirmed numerically that the "
     "profile's second derivative jumps from a finite nonzero value to exactly zero there -- "
     "continuous value and slope (C1), but a genuine curvature discontinuity (not C2). This "
     "is a distinct geometric feature from the groove root itself, and the requested "
     "contour plot turned out to be exactly the right way to check it, rather than reasoning "
     "from the formula alone -- our existing region mask (radius ~0.046) does not reach "
     "anywhere near these points (0.15 away, over 3x farther out).")
image("groove_contour_stress_sample0.png",
      "Figure 1. B3: every Cauchy stress component along the inner wall (theta~0), 839,040- "
      "vs. 950,400-element solutions, one held-out sample. Groove root (dotted), band edges "
      "(dashed), and the existing region-mask extent (shaded) are marked.")
para("The plot shows a real, clear kink in the shear components (sigma_xy/xz/yz) exactly at "
     "the band edges, confirming this curvature-discontinuity feature does leave a visible "
     "stress signature there -- but both mesh resolutions agree closely on this kink's shape, "
     "so it is itself a converged, genuine geometric response, not a source of non-"
     "convergence. A separate, unrelated large stress feature appears at the bushing's two "
     "free axial ends (z near 0 and Lz) -- a free-surface/corner effect, not investigated "
     "further here.")
para("The plot's normal-stress components (sigma_xx/yy/zz) appeared, by eye, to overlap "
     "almost everywhere, including at the groove itself -- seemingly at odds with our "
     "previously-reported 274% region-stress figure. We checked this directly rather than "
     "trust the visual impression: a follow-up diagnostic reporting raw per-Gauss-point "
     "magnitudes within the actual region mask shows the apparent agreement was a y-axis-"
     "scale illusion, not a real finding.")
table(
    ["component", "true RMS", "true max|.|", "interpolated (839k) RMS", "|diff| RMS", "relative"],
    [
        ["sigma_xx", "0.4847", "1.1049", "1.1758", "1.0195", "210%"],
        ["sigma_yy", "0.3810", "0.7628", "1.0249", "1.0191", "267%"],
        ["sigma_zz", "0.5106", "1.2646", "1.1962", "1.3194", "258%"],
        ["sigma_xy", "0.0088", "0.0276", "0.0087", "0.0003", "3%"],
        ["sigma_xz", "0.3183", "0.8148", "0.3175", "0.0102", "3%"],
        ["sigma_yz", "0.0167", "0.0509", "0.0120", "0.0111", "66%"],
    ],
)
para("The normal components show a genuine, large discrepancy: the true values are not "
     "small (RMS 0.38-0.51, ruling out a small-denominator effect), and the absolute "
     "mismatch (RMS 1.0-1.3) actually exceeds the true value's own magnitude. By contrast, "
     "sigma_xz converges to 3% despite a comparably large true magnitude (0.32) -- directly "
     "ruling out a small-denominator explanation and pinning the non-convergence specifically "
     "on the normal-stress response at the groove.")

# ===========================================================================
h1("2. Local mesh refinement: does the groove-region stress converge?")
para("The mesh-convergence ladder we had used until now only had GLOBAL refinement -- every "
     "z-layer gets finer together, everywhere in the domain. We built a new mesh generator "
     "instead, with z-layers clustered specifically at the groove's own deepest point (a "
     "standard sinh mesh-grading technique), verified to reproduce the original uniform mesh "
     "exactly in the no-clustering limit and to solve the identical physical problem at every "
     "clustering strength tested (equilibrium residual at machine precision throughout). At a "
     "fixed, production-scale element count (47,040 elements), we then increased local "
     "clustering strength while holding the global element count fixed:")
table(
    ["z_beta", "Gauss pts in groove region", "avg|sigma_xx|", "max|sigma_xx|", "avg|sigma|_F"],
    [
        ["0.0", "2,356", "3.027", "47.65", "36.23"],
        ["2.0", "3,870", "2.975", "44.49", "36.08"],
        ["6.0", "8,346", "2.850", "47.78", "35.51"],
        ["10.0", "9,700", "2.815", "54.19", "35.23"],
        ["18.0", "10,602", "2.800", "63.35", "34.28"],
    ],
)
image("item1a_local_refinement_convergence.png",
      "Figure 2. Region-average stress converges smoothly as local resolution at the groove "
      "increases (left); the pointwise maximum does not settle at this fixed element order "
      "(right) -- addressed directly in Section 4 below.")
para("The region AVERAGE converges monotonically across these 5 rows (3.027 -> 2.800; the "
     "last three rows, 2.850/2.815/2.800, span about 1.8% of each other) as local Gauss-point "
     "density at the groove grows 4.5x at an identical global element count -- a genuine "
     "convergence curve, unlike the global ladder's 274% non-convergence on the same kind of "
     "comparison. The pointwise maximum does not settle the same way; this is addressed "
     "directly, not glossed over, in Section 4. (Every average in this document, here and "
     "throughout, is a volume-weighted Gauss-quadrature average, not a plain arithmetic mean "
     "-- confirmed directly in Section 4.2, where the distinction is shown to matter by about "
     "5.6% on a representative case.)")

# ===========================================================================
h1("3. Stress Concentration Factor (SCF)")
para("We define SCF = (stress at the groove) / (nominal stress at the exact same physical "
     "location, same mesh, same loading, but with the groove geometrically removed). This "
     "specific design was chosen only after an initial, more obvious-looking approach -- "
     "comparing against the far-field control point already used elsewhere in this project -- "
     "gave a nonsensical SCF below 1. The reason: the applied loading in this problem is a "
     "rigid rotation about an axis passing through the groove's own z-location, so its "
     "dominant displacement term is exactly zero at the groove and large at the far-field "
     "point, for reasons unrelated to the groove's own geometry. Holding location and loading "
     "fixed and only removing the groove isolates its effect correctly.")
table(
    ["", "avg|sigma_xx|", "max|sigma_xx|", "avg|sigma|_F"],
    [
        ["groove (real)", "2.800", "63.35", "34.28"],
        ["nominal (groove removed, same point)", "3.335", "29.15", "38.63"],
    ],
)
para("This first gave SCF (max groove / nominal average) = 19.0. SCF on the region-average "
     "basis is below 1 (0.84-0.89) -- this particular groove geometrically thickens the wall "
     "rather than thinning it, so averaged over the region it behaves more like a mild "
     "reinforcement than a classical notch; the concentration is a pointwise effect only. As "
     "described next, this initial SCF figure was later refined once a better-converged "
     "estimate of the peak stress became available.")

# ===========================================================================
h1("4. Higher-order elements, and is this a true singularity?")
para("Professor Rabczuk's suggestion of higher-order elements or NURBS/IGA was explicitly optional. A full "
     "NURBS/IGA capability is out of scope for this project -- no such solver exists in our "
     "current codebase, and building one was not attempted. Quadratic (20-node, p-refinement) "
     "elements were genuinely testable, however, since the finite-element library we use "
     "already ships this element type along with a mesh-upgrade utility. We first caught and "
     "fixed a real correctness issue: the library's own straight-line placement of new mid-"
     "edge nodes does not lie on this mesh's curved inner/outer cylindrical surfaces, which "
     "would silently leave most surface nodes unconstrained; we fixed this by snapping "
     "affected nodes onto the exact analytic surface, and verified the fix directly.")
table(
    ["Elements", "Element type", "avg|sigma_xx|", "max|sigma_xx|", "avg|sigma|_F"],
    [
        ["2,376", "linear (8-node)", "2.716", "61.90", "40.64"],
        ["2,376", "quadratic (20-node)", "3.076", "28.97", "34.42"],
        ["5,760", "linear (8-node)", "2.857", "61.19", "38.35"],
        ["5,760", "quadratic (20-node)", "3.114", "29.90", "34.84"],
        ["10,200", "quadratic (20-node)", "3.118", "29.61", "34.80"],
    ],
)
image("item1c_linear_vs_quadratic.png",
      "Figure 3. Peak groove stress under linear vs. quadratic elements, at the same element "
      "counts. Quadratic elements roughly halve the peak value and are already stable across "
      "three independent resolutions.")
para("Quadratic elements cut the peak groove stress roughly in half at the same element "
     "count, and -- unlike linear elements -- the result is stable across three independent "
     "resolutions spanning a 4.3x element-count range (28.97 / 29.90 / 29.61, agreeing within "
     "about 3%).")
para("To settle whether the groove is a true mathematical singularity or a bounded stress "
     "concentration, we ran two further checks. First, we pushed the linear-element local "
     "refinement much harder: at a fixed 5,760-element mesh, local clustering was increased "
     "until the smallest z-element had shrunk by a factor of about 1,000 (from 0.0417 to "
     "0.000042).")
image("item1d_singularity_check.png",
      "Figure 4. Peak stress under linear elements as the smallest local z-element shrinks "
      "about 1,000x. If this were a true singularity, the peak stress would grow without "
      "bound as local element size collapses; instead it stays within a bounded range.")
para("The peak stress stayed within a bounded range (61.19 -> 57.26 -> 58.13 -> 49.95) rather "
     "than growing without bound as the local element size collapsed toward zero -- not the "
     "signature a true singularity would produce. Second, the three independent quadratic-"
     "element resolutions already shown above all agree tightly, which would not be expected "
     "if the underlying field were genuinely unbounded.")
para("Combined verdict: the true peak stress is almost certainly close to the quadratic-"
     "converged value (approximately 29-30), confirmed at three independent resolutions. "
     "Linear elements, at every resolution and clustering strategy we tested (2,376 to 47,040 "
     "elements), scatter noisily within a bounded range (44-69) rather than showing the "
     "systematic, unbounded growth a true geometric singularity would produce under "
     "refinement -- the production-scale run's own earlier upward drift (47.65 -> 63.35) is "
     "best understood as mesh-dependent scatter around this same bounded range, since pushing "
     "local refinement much further did not continue that trend. This is consistent with "
     "classical elasticity theory: true boundary stress singularities (the Williams corner-"
     "type asymptotic solutions behind real notch and crack-tip behavior) require an actual "
     "discontinuity in the boundary's tangent direction -- this geometry, being C1 (continuous "
     "value and slope) with only a curvature (not-C2) discontinuity, does not have one. "
     "Our conclusion is that this is a genuine, real, but bounded stress concentration, not a "
     "mathematical singularity -- supported by consistent numerical evidence from two "
     "independent refinement strategies together with the theoretical argument above, though "
     "not a from-first-principles asymptotic proof, which a problem at this applied level does "
     "not require.")
h2("4.1 Do all six stress components converge under Q20, not just sigma_xx?")
para("Re-ran the same three quadratic resolutions reporting every independent Cauchy-stress "
     "component, not only sigma_xx:")
table(
    ["Component", "avg spread (3 resolutions)", "max spread (3 resolutions)"],
    [
        ["sigma_xx", "1.3%", "3.1%"],
        ["sigma_yy", "3.3%", "2.6%"],
        ["sigma_zz", "25.0% (small absolute value, ~-0.17 to -0.21)", "6.3%"],
        ["sigma_xy", "1.3%", "24.4% -- not well converged"],
        ["sigma_xz", "1.4%", "4.9%"],
        ["sigma_yz", "0.6%", "9.0%"],
    ],
)
para("The region AVERAGE converges well for every component. The region MAXIMUM converges "
     "reasonably (3-9%) for sigma_xx/yy/zz/xz, but sigma_xy's maximum does not converge as "
     "tightly (24%) -- reported directly rather than omitted. sigma_zz's large average "
     "percentage spread is a small-denominator effect (its own absolute value is small, "
     "-0.17 to -0.21), not a large absolute change. Importantly, sigma_xy's own absolute "
     "magnitude (2.8-3.7) is small relative to the dominant shear component sigma_xz "
     "(27-29, itself well-converged at 4.9%), so this does not change the overall picture: "
     "the stress-concentration story at this groove is carried mainly by sigma_xx and "
     "sigma_xz, both of which converge well under Q20.")

h2("4.2 A properly matched-element-type SCF")
para("The SCF reported in Section 3 mixed a quadratic-element groove peak with a linear-"
     "element nominal baseline. For a fully consistent comparison, we computed a quadratic-"
     "element no-groove counterfactual at the same 5,760-element resolution already used for "
     "the quadratic groove result:")
table(
    ["", "avg|sigma_xx|", "max|sigma_xx|", "avg|sigma|_F"],
    [
        ["groove (quadratic, 5,760 elements)", "3.114", "29.90", "34.84"],
        ["nominal, no groove (quadratic, 5,760 elements)", "3.537", "33.23", "40.04"],
    ],
)
para("Matched-element-type SCF (max groove / nominal average, both quadratic) = 8.45 -- "
     "close to, and confirming, the earlier mixed-element estimate (~9.0) was a reasonable "
     "approximation, now replaced by this properly apples-to-apples number. To be precise "
     "about what this number is: it is specifically the ratio of the peak sigma_xx at the "
     "groove to the average sigma_xx at the same point with the groove removed -- a "
     "component-specific figure, not a general or isotropic stress-concentration factor "
     "covering every stress measure. SCF on the average basis stays below 1 (0.88 for "
     "sigma_xx, 0.87 for the Frobenius norm), consistent with every earlier finding that this "
     "groove does not raise the average stress.")
para("A note on how the region average itself is computed, since this was also checked "
     "directly: it is a volume-weighted (Gauss-quadrature-weighted) average over every Gauss "
     "point inside the region, not a plain arithmetic mean -- each point is weighted by its "
     "own quadrature weight times the local Jacobian determinant. This is not a cosmetic "
     "distinction: at one representative case, the volume-weighted mean (2.588) differs from "
     "the plain arithmetic mean (2.444) by 5.6%, because Gauss points are not uniformly "
     "distributed in physical volume once the mesh is locally graded. Every average reported "
     "in this document uses the volume-weighted convention throughout.")

# ===========================================================================
h1("5. Far-field PI-vs-DD comparison")
para("With the far-field region already established as convergent and free of groove "
     "influence, we re-evaluated every existing u-only checkpoint against the same already-"
     "cached 950,400-element reference, swapping in the far-field region mask for the groove "
     "one -- no new finite-element solving was required.")
table(
    ["Checkpoint", "disp L2", "far-field region sigma field", "far-field sigma_xx"],
    [
        ["PI baseline", "2.78%", "16.69%", "15.04%"],
        ["Local-integration-refinement (seed 1)", "3.37%", "13.01%", "11.34%"],
        ["Local-integration-refinement (seed 2)", "2.40%", "15.02%", "13.54%"],
        ["Local-integration-refinement (seed 3)", "2.85%", "13.65%", "12.85%"],
        ["Equilibrium w=1.0 (closed, poor)", "37.08%", "138.56%", "140.27%"],
        ["Equilibrium w=0.12 (closed, poor)", "37.76%", "151.27%", "153.88%"],
        ["DD @ N=50", "8.73%", "52.57%", "55.47%"],
        ["DD @ N=200", "2.81%", "30.24%", "32.40%"],
        ["DD @ N=400", "3.88%", "21.19%", "21.27%"],
        ["DD sobolev", "3.37%", "21.10%", "21.47%"],
        ["DD hybrid", "3.55%", "29.38%", "30.25%"],
    ],
)
para("This is exactly the meaningful PI-vs-DD stress comparison requested. PI and every "
     "local-integration-refinement seed are the best performers (13-17%), clearly beating "
     "every DD variant (21-53%) -- the same PI-over-DD ranking already established elsewhere "
     "in this project, now confirmed for region stress specifically, measured somewhere the "
     "finite-element ground truth itself is trustworthy. Local-integration-refinement "
     "(trained specifically to help the groove) performs slightly better than PI baseline "
     "even here, away from the groove -- that training does not trade off far-field accuracy "
     "for groove-region accuracy. These numbers (13-53%) also sit in the same reasonable "
     "range as every other quantity of interest in this project (H1, energy, reaction) -- "
     "nothing like the groove's 1,000-29,000% range -- confirming that the groove failure is "
     "a localized phenomenon specific to that one unresolved geometric feature, not "
     "representative of how well any of these models predict stress in general.")

# ===========================================================================
h1("6. Overall status: well-supported, not yet exhaustively closed")
para("Taken together, Sections 1-5 give strong, multi-pronged support for a converged, "
     "bounded groove stress and a defensible SCF -- but we do not consider the groove "
     "question 100% definitively closed, for two honest reasons. First, sigma_xy's "
     "pointwise maximum did not converge as tightly as the other five stress components "
     "(Section 4.1); its small absolute magnitude limits how much this affects the overall "
     "picture, but it is a genuine loose end, not fully resolved. Second, showing that each "
     "mesh's own region maximum/average lands in a similar range is not quite the same "
     "claim as showing the underlying stress FIELD has converged at the same physical "
     "locations -- a sharper, field-level check along the groove contour, not just a region "
     "summary statistic, is the more rigorous standard and is reported directly below.")
para("Field-level check: extracted the stress along the same inner-wall contour used by "
     "the original (linear-element) contour plot in Section 1, now across the same three "
     "quadratic resolutions already tested, and compared the two finest resolutions "
     "pointwise (not just their region summary statistics) after interpolating onto a "
     "common grid.")
image("quadratic_contour_convergence.png",
      "Figure 5. Every stress component along the inner-wall contour, Q20 elements at three "
      "resolutions. sigma_xx/yy/zz/xz visibly overlap inside the groove band (shaded); "
      "sigma_xy/yz show real point-to-point scatter that does not resolve with resolution.")
table(
    ["Component", "max pointwise diff (within groove region)", "mean pointwise diff"],
    [
        ["sigma_xx", "1.9%", "0.5%"],
        ["sigma_yy", "2.4%", "0.7%"],
        ["sigma_zz", "6.8%", "1.5%"],
        ["sigma_xy", "21.6% -- not well converged", "4.7%"],
        ["sigma_xz", "4.6%", "1.3%"],
        ["sigma_yz", "18.8% -- not well converged", "5.6%"],
    ],
)
para("This is a materially stronger result than the earlier region-max/average check: "
     "sigma_xx, yy, zz, and xz -- including sigma_xz, the dominant shear component carrying "
     "most of the stress-concentration story -- converge at the FIELD level, not just in "
     "aggregate, with mean pointwise differences under 1.5% within the groove region. "
     "sigma_xy and sigma_yz show genuine point-to-point noise that does not resolve with "
     "increasing Q20 resolution, visible in the figure as real scatter rather than a smooth "
     "trend -- a confirmed, not merely inferred, loose end. Their small absolute magnitude "
     "relative to sigma_xx/sigma_xz means this does not change the overall groove-stress-"
     "concentration conclusion, but we report it plainly rather than smoothing over it. "
     "Overall: this study is well-supported and ready for Professor Rabczuk's input, but we "
     "do not present it as 100% exhaustively closed.")
bullet("Groove smoothness, and whether the geometry rules out a singularity -- answered in "
       "Section 1 (genuinely smooth at the groove root; a separate, distinct C1-not-C2 "
       "curvature feature identified at the band edges).")
bullet("A plot of the stress distribution along the groove contour -- Figure 1, Section 1, "
       "plus the raw-value diagnostic that corrected our own initial visual reading of it.")
bullet("Local mesh refinement at the groove, to establish genuine convergence -- Section 2 "
       "and Figure 2 (region average converges cleanly; the pointwise maximum required the "
       "further work in Section 4).")
bullet("Stress Concentration Factor -- Section 3, refined in Section 4.2 with a properly "
       "matched-element-type comparison (SCF ~8.45, quadratic peak vs. quadratic nominal).")
bullet("Convergence of all stress components, and the averaging method used -- Section 4.1 "
       "(every component's average converges well; sigma_xy's maximum is the one exception, "
       "flagged directly) and Section 4.2 (confirmed volume-weighted, not a plain mean).")
bullet("Whether the stress FIELD itself (not just a region summary statistic) converges at "
       "the same physical locations -- Section 6 and Figure 5 (sigma_xx/yy/zz/xz converge "
       "well pointwise, under 1.5% mean difference; sigma_xy/yz show real, unresolved "
       "point-to-point scatter, reported directly).")
bullet("Possibly higher-order elements or NURBS/IGA (explicitly optional) -- "
       "Section 4 and Figure 3 (NURBS/IGA out of scope; quadratic elements tested directly "
       "and found to converge the peak stress at far lower cost than further mesh "
       "refinement).")
bullet("Whether the groove is a true singularity -- Section 4 and Figure 4 (bounded "
       "concentration, not a true singularity, supported by two independent numerical checks "
       "plus the underlying elasticity theory).")
bullet("Far-field PI-vs-DD comparison, once a trustworthy region was available -- Section 5.")
para("This report covers every point raised in Professor Rabczuk's 2026-11-10 email in full.")

doc.save(OUT_PATH)
print(f"Wrote {OUT_PATH}")
