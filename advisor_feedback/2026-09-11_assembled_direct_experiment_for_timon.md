# Update on GPU-FEM: a faster solver variant (see attached files) — drafted 2026-09-11

Do not send without Omar's own review. Kept deliberately SHORT per
Omar's own direct instruction ("انا الايميل بدي يكون عام والنصوص
والجداول تكون في الملفات" — the email should be general, with the text
and tables in the files) — full numbers, methodology, and verification
detail for Points 8 and 9 live in `PFEM_Transolver_Report_2026-09-09.docx`
and `PFEM_Work_Summary_2026-09-09.docx` (both now have real Word tables
for this content, not just prose), not here.

---

Subject: Update on GPU-FEM — a faster solver variant (see attached files)

Dear Professor Rabczuk,

Attached are the updated Report and Summary. Beyond the round-9 points,
I have added two further points (8 and 9) documenting a new experiment
on our own GPU solver.

Point 8: rather than keep our solver matrix-free unconditionally, I
built and tested an assembled-and-direct-solved variant — the same
approach torch-fem and TensorMesh already use. At every resolution
tested so far (N=401 through N=2001), it matches their accuracy while
being faster and using less peak memory than both.

Point 9: a further round of engineering on that same solver (reusing
the direct solver's own matrix reordering across Newton iterations, and
switching to a symmetric matrix factorization) gave an additional,
smaller speedup and a further reduction in peak memory.

Full numbers, methodology, and the verification behind each claim are
in the attached files, under Points 8 and 9 — I have kept this email
short and put the detail there rather than here.

Neither point is a finalized result on our side yet. I would like your
read on whether the underlying approach is sound before it is treated
as more than a promising experiment, and how you would want it
reflected in the report relative to the matrix-free solver already
there.

Best regards,

Omar
