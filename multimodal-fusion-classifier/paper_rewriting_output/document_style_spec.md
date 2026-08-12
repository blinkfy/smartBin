# Word Design Specification

- Base preset: `narrative_proposal`
- Named override: `academic_a4_zh`
- Purpose: Chinese computer-vision journal manuscript

| Component | Resolved Token |
|---|---|
| Page | A4 portrait, 24 mm top/bottom, 25 mm left/right |
| Body | SimSun for CJK, Times New Roman for Latin, 10.5 pt, justified, 1.25 line spacing, 6 pt after, 21 pt first-line indent |
| Title | SimHei/Times New Roman, 18 pt, bold, centered, 12 pt after |
| Heading 1 | SimHei/Times New Roman, 14 pt, bold, black, 14 pt before, 6 pt after |
| Heading 2 | SimHei/Times New Roman, 12 pt, bold, black, 10 pt before, 4 pt after |
| Heading 3 | SimHei/Times New Roman, 11 pt, bold, black, 8 pt before, 4 pt after |
| Caption | SimSun/Times New Roman, 9 pt, centered, 4 pt before/after |
| Tables | 9072 DXA total width, 120 DXA indent, fixed layout, content-specific columns, 80/80/120/120 DXA cell margins, light-gray header |
| Footer | Centered automatic page number, 9 pt Times New Roman |
| Figures | Centered, maximum width 156 mm, aspect ratio preserved |

The A4 page, Chinese font and black-heading choices are named overrides to the
base preset because the target is a Chinese academic manuscript rather than a
business proposal. No decorative cover, colored heading system or running
header is used.

