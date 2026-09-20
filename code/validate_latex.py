"""Structural validation of the LaTeX section (no LaTeX distribution needed)."""
import os
import re
from collections import Counter

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "latex",
                   "results_section.tex")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def strip_comments(src):
    out_lines = []
    for line in src.split("\n"):
        buf, i = [], 0
        while i < len(line):
            if line[i] == "\\":                 # escaped char, keep both
                buf.append(line[i:i + 2])
                i += 2
                continue
            if line[i] == "%":                  # real comment start
                break
            buf.append(line[i])
            i += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


def main():
    src = open(SRC, encoding="utf-8").read()
    body = strip_comments(src)
    ok = True

    print("=== environnements ===")
    b = re.findall(r"\\begin\{([^}]+)\}", body)
    e = re.findall(r"\\end\{([^}]+)\}", body)
    cb, ce = Counter(b), Counter(e)
    for k in sorted(set(cb) | set(ce)):
        flag = "" if cb[k] == ce[k] else "   <-- DESEQUILIBRE"
        if flag:
            ok = False
        print(f"  {k:<14} begin={cb[k]:<3} end={ce[k]:<3}{flag}")

    stack, errs = [], []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        else:
            if not stack or stack[-1] != m.group(2):
                errs.append((m.group(2), stack[-1] if stack else None))
            else:
                stack.pop()
    print("  imbrication :", "OK" if not errs and not stack
          else f"ERREUR {errs[:3]} reste={stack[:3]}")
    if errs or stack:
        ok = False

    print("\n=== accolades ===")
    nb, ne = body.count("{"), body.count("}")
    print(f"  ouvrantes={nb}  fermantes={ne}  ->", "OK" if nb == ne else "DESEQUILIBRE")
    if nb != ne:
        ok = False

    print("\n=== labels / refs ===")
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body))
    print(f"  {len(labels)} labels, {len(refs)} references distinctes")
    missing = sorted(refs - labels)
    unused = sorted(labels - refs)
    print("  refs sans label  :", missing or "aucune")
    print("  labels non cites :", unused or "aucun")
    if missing:
        ok = False

    print("\n=== figures referencees ===")
    gs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    for g in gs:
        exists = os.path.exists(os.path.join(FIG, g))
        print(f"  {g:<38} {'OK' if exists else 'FICHIER MANQUANT'}")
        if not exists:
            ok = False
    print(f"  -> {len(gs)} figures incluses")

    print("\n=== tableaux / citations ===")
    tabs = re.findall(r"\\begin\{table\}", body)
    print("  environnements table :", len(tabs))
    print("  multirow             :", len(re.findall(r"\\multirow", body)))
    print("  captions             :", len(re.findall(r"\\caption", body)))
    cites = set(re.findall(r"\\cite\{([^}]+)\}", body))
    print("  cles citees          :", sorted(cites) or "aucune")

    print("\n=== RESULTAT ===")
    print("  ", "VALIDATION STRUCTURELLE OK" if ok else "PROBLEMES DETECTES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
