import xml.etree.ElementTree as etree
import xml.etree.ElementTree as ET
from xml.dom import minidom

import os, re

METHOD_REGEX = re.compile(
    r"""
    ^\s*                                        # ⬅ anchor to line start
    (?:(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default)\s+)*
    (?:<[^>;{()]*>\s*)?
    (?!class\b|interface\b|enum\b|record\b)
    (?:
        [A-Za-z_][$\w]* (?:\s*\.\s*[A-Za-z_][$\w]*)*
        (?:\s*<[^>;{}()]*>)?
        (?:\s*\[\s*\])*
        | void
    )
    \s+
    ([A-Za-z_][$\w]*)                            # method name
    \s*
    \(
        [^()]* (?:\([^()]*\)[^()]*)*
    \)
    \s*
    (?:throws\s+[A-Za-z_][$\w]*(?:\s*\.\s*[A-Za-z_][$\w]*)*(?:\s*,\s*[A-Za-z_][$\w]*(?:\s*\.\s*[A-Za-z_][$\w]*)*)*)?
    \s*
    (?:\{|\;)                                    # body or abstract/interface
    """,
    re.VERBOSE | re.MULTILINE
)

LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
STRING_LIT  = re.compile(r"\"(?:\\.|[^\"\\])*\"|\'(?:\\.|[^\'\\])*\'")

def _strip_comments_and_strings(code: str) -> str:
    code = STRING_LIT.sub('""', code)
    code = BLOCK_COMMENT.sub("", code)
    code = LINE_COMMENT.sub("", code)
    return code

def count_java_methods_in_file(java_file_path: str) -> int:
    if not (java_file_path.lower().endswith(".java") and os.path.isfile(java_file_path)):
        raise ValueError("Please provide a valid path to a .java file.")
    with open(java_file_path, "r", encoding="utf-8", errors="ignore") as fh:
        code = fh.read()
    cleaned = _strip_comments_and_strings(code)
    return sum(1 for _ in METHOD_REGEX.finditer(cleaned))


# Output functions
def printWarning(message):
    print('\033[93m WARNING: ' + message + '\033[0m')

def printError(message):
    print('\033[91m ERROR: ' + message + '\033[0m')

def printInfo(message):
    print('\033[96m INFO: ' + message + '\033[0m')

# Classes
class CloneFragment():
    def __init__(self, file, ls, le, fn = "", fh = 0):
        self.file = file
        self.ls = ls
        self.le = le
        self.function_name = fn
        self.function_hash = fh

    def contains(self, other):
        return self.file == other.file and self.ls <= other.ls and self.le >= other.le

    def __eq__(self, other):
        return self.file == other.file and self.ls == other.ls and self.le == other.le

    def matches(self, other):
        return self.file == other.file and self.function_name == other.function_name

    def matchesStrictly(self, other):
        return self.file == other.file and self.function_name == other.function_name and (self.ls == other.ls or self.function_hash == other.function_hash)

    def __hash__(self):
        return hash(self.file+str(self.ls))

    def toXML(self):
        return "\t\t\t<source file=\"%s\" startline=\"%d\" endline=\"%d\" function=\"%s\" hash=\"%d\"></source>\n" % (self.file, self.ls, self.le,self.function_name, self.function_hash)

class CloneClass():
    def __init__(self):
        self.fragments = []

    def contains(self, fragment):
        for f in self.fragments:
            if f.matches(fragment):
                return True
        return False

    def toXML(self):
        s = "\t\t<class nclones=\"%d\">\n" % (len(self.fragments))
        for fragment in self.fragments:
            s += fragment.toXML()
        s += "\t\t</class>\n"
        return s

class CloneVersion():
    def __init__(self, cc, h, n, evo = "None", chan = "None"):
        self.cloneclass = cc
        self.hash = h
        self.nr = n
        self.evolution_pattern = evo
        self.change_pattern = chan

    def toXML(self):
        s = "\t<version nr=\"%d\" hash=\"%s\" evolution=\"%s\" change=\"%s\">\n" % (self.nr, self.hash, self.evolution_pattern, self.change_pattern)
        s += self.cloneclass.toXML()
        s += "\t</version>\n"
        return s

    def getMatchScore(self, version):
        score = 0
        for fragment1 in self.cloneclass.fragments:
            found = False
            for fragment2 in version.cloneclass.fragments:
                if fragment1.matchesStrictly(fragment2):
                    found = True
                    break
            if found:
                score += 2
            else:
                for fragment2 in version.cloneclass.fragments:
                    if fragment1.matches(fragment2):
                        score += 1
                        break
        return score

class Lineage():
    def __init__(self):
        self.versions = []

    def matches(self, cc):
        for fragment in cc.fragments:
            if self.versions[-1].cloneclass.contains(fragment):
                return True
        return False

    def containsInconsistentChange(self):
        for version in self.versions:
            if version.change_pattern == "Inconsistent":
                return True
        return False

    def countInconsistentChange(self):
        count = 0
        for version in self.versions:
            if version.change_pattern == "Inconsistent":
                count+=1
        return count

    def containsConsistentChange(self):
        for version in self.versions:
            if version.change_pattern == "Consistent":
                return True
        return False

    def countConsistentChange(self):
        count = 0
        for version in self.versions:
            if version.change_pattern == "Consistent":
                count+=1
        return count

    def containsDoubleVersions(self):
        n_prev = -1
        for version in self.versions:
            if version.nr == n_prev:
                return True
            else:
                n_prev = version.nr
        return False

    def getLength(self):
        if len(self.versions) == 1:
            return 1
        return (self.versions[-1].nr - self.versions[0].nr)

    def toXML(self):
        s = "<lineage>\n"
        for version in self.versions:
            s += version.toXML()
        s += "</lineage>\n"
        return s

def getLastCommitFromDensityCSV(filename):
    try:
        with open(filename, 'r+') as file:
            return int(file.readlines()[-1].split(",")[0])
    except Exception as e:
        printError("Something went wrong while parsing the density dataset:")
        printError(str(e))
        return 0

def parseLineageFile(filename):
    lineages = []
    with open(filename, 'r', encoding='utf-8') as file:
        try:
            file_xml = etree.parse(file)
            root = file_xml.getroot()

            for lineage_el in root:
                lin = Lineage()

                # versões são os filhos diretos de lineage_el
                for version_el in lineage_el:  # equivalente a: for version_el in list(lineage_el):
                    cc = CloneClass()

                    # cloneclasses são os filhos de version_el
                    cloneclasses = list(version_el)
                    if len(cloneclasses) != 1:
                        printWarning("Unexpected amount of clone classes in version.")
                        printWarning("Please check if inputfile is consistent.")

                    if cloneclasses:
                        # fragments são os filhos do primeiro cloneclass
                        for fragment_el in cloneclasses[0]:
                            cc.fragments.append(
                                CloneFragment(
                                    fragment_el.get("file"),
                                    int(fragment_el.get("startline")),
                                    int(fragment_el.get("endline")),
                                    fragment_el.get("function"),
                                    int(fragment_el.get("hash"))
                                )
                            )

                    cv = CloneVersion(
                        cc,
                        version_el.get("hash"),
                        int(version_el.get("nr")),
                        version_el.get("evolution"),
                        version_el.get("change")
                    )
                    lin.versions.append(cv)

                lineages.append(lin)

        except Exception as e:
            printError("Something went wrong while parsing the lineage dataset:")
            printError(str(e))
            return []

    return lineages

def getNrOfInconsistentChangeLineages(lineages):
    count = 0
    for lineage in lineages:
        if lineage.containsInconsistentChange():
            count+=1
    return count

def getNrOfConsistentChangeLineages(lineages):
    count = 0
    for lineage in lineages:
        if lineage.containsConsistentChange() and not lineage.containsInconsistentChange():
            count+=1
    return count

def examineChangeLineages(lineages):
    consistent = []
    inconsistent = []
    inconsistent2 = []
    for lineage in lineages:
        if lineage.containsConsistentChange() and not lineage.containsInconsistentChange():
            consistent.append(lineage.countConsistentChange())
        elif lineage.containsInconsistentChange():
            inconsistent.append(lineage.countConsistentChange())
            inconsistent2.append(lineage.countInconsistentChange())
    print("CONSISTENT CHANGES IN CONSISTENT LINEAGE:" + overviewStringList(consistent))
    print("CONSISTENT CHANGES IN INCONSISTENT LINEAGE:" + overviewStringList(inconsistent))
    print("INCONSISTENT CHANGES IN INCONSISTENT LINEAGE:" + overviewStringList(inconsistent2))
    print("")

def getNrOfStableLineages(lineages):
    count = 0
    for lineage in lineages:
        if not lineage.containsConsistentChange() and not lineage.containsInconsistentChange():
            count+=1
    return count

def getNrOfAliveLineages(lineages, last_commit_nr):
    count = 0
    for lineage in lineages:
        if lineage.versions[-1].nr == last_commit_nr:
            count+=1
    return count

def getLenghtsOfDeadLineages(lineages, last_commit_nr):
    lengths = []
    for lineage in lineages:
        if lineage.versions[-1].nr != last_commit_nr:
            lengths.append(lineage.getLength())
    return lengths

def countVersions(lineages):
    data = {}
    data["total"] = 0
    data["Consistent"] = 0
    data["Inconsistent"] = 0
    data["Same"] = 0
    data["Add"] = 0
    data["Subtract"] = 0
    for lineage in lineages:
        last_nr = 0
        for version in lineage.versions:
            if version.evolution_pattern == "None" and version.change_pattern == "None":
                last_nr = version.nr
            elif version.evolution_pattern == "Same" and version.change_pattern == "Same":
                num = (version.nr - last_nr)/30
                data[version.evolution_pattern] += num
                data[version.change_pattern] += num
                data["total"] += num
                last_nr = version.nr
            else:
                data[version.evolution_pattern] += 1
                data[version.change_pattern] += 1
                data["total"] += 1
                last_nr = version.nr
    return data

def percentageString(part,total):
    return str(part) + " (" + str((100*part)/total) + "%)"

def percentageStringLatex(part,total):
    try:
        return "%0.2f" % ((100*part)/total)
    except:
        return "0.00"

def overviewString2(part1, part2, total, legend):
    return percentageString(part1, total) + " / " + percentageString(part2, total) + " (" + legend + ")"

def overviewString2Latex(part1, part2, total, legend):
    # FIXED: escape % for Python strings / LaTeX
    return (percentageStringLatex(part1, total) + " \\% & " +
            percentageStringLatex(part2, total) + "\\% (" + legend + ")")

def overviewString3(part1, part2, part3, total, legend):
    return percentageString(part1, total) + " / " + percentageString(part2, total) + " / "+  percentageString(part3, total) + " (" + legend + ")"

def overviewString3Latex(part1, part2, part3, total, legend):
    # FIXED: escape % for Python strings / LaTeX
    return (percentageStringLatex(part1, total) + "\\% & " +
            percentageStringLatex(part2, total) + "\\% & " +
            percentageStringLatex(part3, total) + "\\% (" + legend + ")")

def overviewStringList(l):
    if len(l) == 0:
        return "0" + " / " + "0" + " / " + "0" + " (min/avg/max)"
    return str(min(l)) + " / " + str(sum(l)/len(l)) + " / " + str(max(l)) + " (min/avg/max)"

def recalculate_lineage(lineage):
    for i in range(len(lineage.versions)):
        cur_n = lineage.versions[i].nr
        prev_n = -1
        j = i - 1
        while (j >= 0 and cur_n == prev_n):
            prev_n = lineage.versions[j].nr
            j-=1
        if j != -1:
            # print("REANALYZING: ")
            # print(lineage.versions[i].toXML())
            best_match = lineage.versions[j]
            score = lineage.versions[i].getMatchScore(lineage.versions[j])
            while (j >= 0 and lineage.versions[j].nr == prev_n):
                # print(" >>> POSSIBLE MATCH:")
                # print(lineage.versions[j].toXML())
                if score < lineage.versions[i].getMatchScore(lineage.versions[j]):
                    score = lineage.versions[i].getMatchScore(lineage.versions[j])
                    best_match = lineage.versions[j]
                j-=1
            # print(" <<< FINAL MATCH:")
            # print(best_match.toXML())
            # evolution
            if len(best_match.cloneclass.fragments) == len(lineage.versions[i].cloneclass.fragments):
                lineage.versions[i].evolution_pattern = "Same"
            elif len(best_match.cloneclass.fragments) > len(lineage.versions[i].cloneclass.fragments):
                lineage.versions[i].evolution_pattern = "Subtract"
            else:
                # print("CHANGING TO ADD FROM " + lineage.versions[i].evolution_pattern)

                lineage.versions[i].evolution_pattern = "Add"


            # change
            if score == 2*len(lineage.versions[i].cloneclass.fragments):
                lineage.versions[i].change_pattern = "Same"
            elif score == len(lineage.versions[i].cloneclass.fragments):
                lineage.versions[i].change_pattern = "Consistent"
            else:
                lineage.versions[i].change_pattern = "Inconsistent"

def Analysis(RES_DIR):
    print("STARTING CLONE GENEOLOGY ANALYSIS\n")
    print(RES_DIR)
    # Files
    P_RES_FILE = RES_DIR + "/production_results.xml"
    P_DENS_FILE = RES_DIR + "/production_density.csv"

    print(" Importing data from " + P_RES_FILE)
    P_LIN_DATA = parseLineageFile(P_RES_FILE)
    if not len(P_LIN_DATA):
        printError("Empty production data: no linages found in " + P_RES_FILE)
        return

    p_last_commit = getLastCommitFromDensityCSV(P_DENS_FILE)
    print("  >> All data imported\n")

    for lineage in P_LIN_DATA:
        if lineage.containsDoubleVersions():
            recalculate_lineage(lineage)

    print("\n--- RESULTS ---\n")

    print("Total amount of clone lineages")
    print(" - production: " + str(len(P_LIN_DATA)))

    print("\nChange patterns of lineages:")
    p_inconsistent = getNrOfInconsistentChangeLineages(P_LIN_DATA)
    p_consistent = getNrOfConsistentChangeLineages(P_LIN_DATA)
    p_stable = getNrOfStableLineages(P_LIN_DATA)
    print(" - production: " + overviewString3Latex(p_consistent, p_stable, p_inconsistent, len(P_LIN_DATA), "consistent/stable/inconsistent"))

    print("\nStatus of clone lineages:")
    p_alive = getNrOfAliveLineages(P_LIN_DATA, p_last_commit)
    p_dead = len(P_LIN_DATA) - p_alive
    print(" - production: " + overviewString2(p_alive, p_dead, len(P_LIN_DATA), "alive/dead"))

    print("\nLength of dead clone lineages:")
    p_dead_length = getLenghtsOfDeadLineages(P_LIN_DATA, p_last_commit)
    print(" - production: " + overviewStringList(p_dead_length))
    p_dead_length.sort()
    print(" - production full list:" + str(p_dead_length))

    print("\nTotal amount of versions:")
    p_data = countVersions(P_LIN_DATA)
    print(" - production: " + str(p_data["total"]))

    print("\nEvolution pattern of versions:")
    p_same_evo = p_data["total"] - p_data["Add"] - p_data["Subtract"]
    print(" - production: " + overviewString2Latex(p_data["Add"], p_data["Subtract"], p_data["Add"]+ p_data["Subtract"], "add/subtract"))

    print("\nChange pattern of versions:")
    p_same_change = p_data["total"] - p_data["Consistent"] - p_data["Inconsistent"]
    print(" - production: " + overviewString2Latex(p_data["Consistent"], p_data["Inconsistent"], p_data["Consistent"] + p_data["Inconsistent"] , "consistent/inconsistent"))
    examineChangeLineages(P_LIN_DATA)

    if (p_same_evo + p_same_change != p_data["Same"]):
        printWarning("Calculated amount of same versions does not match measured amount in production data.")

    print("\nDONE")

def Analysis(RES_DIR):
    P_RES_FILE = RES_DIR + "/production_results.xml"
    P_DENS_FILE = RES_DIR + "/production_density.csv"

    P_LIN_DATA = parseLineageFile(P_RES_FILE)
    if not len(P_LIN_DATA):
        printError("Empty production data: no linages found in " + P_RES_FILE)
        return None

    p_last_commit = getLastCommitFromDensityCSV(P_DENS_FILE)

    for lineage in P_LIN_DATA:
        if lineage.containsDoubleVersions():
            recalculate_lineage(lineage)

    total_lineages = len(P_LIN_DATA)

    p_inconsistent = getNrOfInconsistentChangeLineages(P_LIN_DATA)
    p_consistent = getNrOfConsistentChangeLineages(P_LIN_DATA)
    p_stable = getNrOfStableLineages(P_LIN_DATA)

    p_alive = getNrOfAliveLineages(P_LIN_DATA, p_last_commit)
    p_dead = total_lineages - p_alive

    p_dead_length = getLenghtsOfDeadLineages(P_LIN_DATA, p_last_commit)
    if p_dead_length:
        dead_min = min(p_dead_length)
        dead_avg = float(sum(p_dead_length)) / float(len(p_dead_length))
        dead_max = max(p_dead_length)
    else:
        dead_min = 0
        dead_avg = 0
        dead_max = 0

    p_data = countVersions(P_LIN_DATA)
    add_ = p_data.get("Add", 0)
    sub_ = p_data.get("Subtract", 0)
    cons_v = p_data.get("Consistent", 0)
    incons_v = p_data.get("Inconsistent", 0)
    total_versions = p_data.get("total", 0)

    same_evo = total_versions - add_ - sub_
    same_change = total_versions - cons_v - incons_v
    if (same_evo + same_change) != p_data.get("Same", 0):
        printWarning("Calculated amount of same versions does not match measured amount in production data.")

    def pct(part, whole):
        return 0.0 if not whole else float(part) * 100.0 / float(whole)

    add_sub_total = add_ + sub_
    cons_incons_total = cons_v + incons_v

    root = ET.Element("metrics")

    def add_node(parent, tag, text=None):
        el = ET.SubElement(parent, tag)
        if text is not None:
            el.text = str(text)
        return el

    add_node(root, "total_clone_lineages", total_lineages)

    cpol = add_node(root, "change_patterns_of_lineages")
    add_node(cpol, "consistent", round(pct(p_consistent, total_lineages), 2))
    add_node(cpol, "stable", round(pct(p_stable, total_lineages), 2))
    add_node(cpol, "inconsistent", round(pct(p_inconsistent, total_lineages), 2))

    status = add_node(root, "status_of_clone_lineages")
    alive = add_node(status, "alive")
    add_node(alive, "count", p_alive)
    add_node(alive, "percentage", pct(p_alive, total_lineages))
    dead = add_node(status, "dead")
    add_node(dead, "count", p_dead)
    add_node(dead, "percentage", pct(p_dead, total_lineages))

    ldead = add_node(root, "length_of_dead_clone_lineages")
    add_node(ldead, "min", dead_min)
    add_node(ldead, "avg", dead_avg)
    add_node(ldead, "max", dead_max)

    add_node(root, "total_amount_of_versions", total_versions)

    evo = add_node(root, "evolution_pattern_of_versions")
    add_node(evo, "add", round(pct(add_, add_sub_total), 2) if add_sub_total else 0.0)
    add_node(evo, "subtract", round(pct(sub_, add_sub_total), 2) if add_sub_total else 0.0)

    chg = add_node(root, "change_pattern_of_versions")
    add_node(chg, "consistent", round(pct(cons_v, cons_incons_total), 2) if cons_incons_total else 0.0)
    add_node(chg, "inconsistent", round(pct(incons_v, cons_incons_total), 2) if cons_incons_total else 0.0)

    ccc = add_node(root, "consistent_changes_in_consistent_lineage")
    add_node(ccc, "min", 0)
    add_node(ccc, "avg", 0)
    add_node(ccc, "max", 0)

    cci = add_node(root, "consistent_changes_in_inconsistent_lineage")
    add_node(cci, "min", 0)
    add_node(cci, "avg", 0)
    add_node(cci, "max", 0)

    iii = add_node(root, "inconsistent_changes_in_inconsistent_lineage")
    add_node(iii, "min", 0)
    add_node(iii, "avg", 0)
    add_node(iii, "max", 0)

    try:
        ET.indent(root, space="  ", level=0)
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        return xml_bytes.decode("utf-8")
    except AttributeError:
        dom = minidom.parseString(rough)
        return dom.toprettyxml(encoding="utf-8").decode("utf-8")


def generateCloneLengthFiles(RES_DIR):
        print("STARTING CLONE LENGTH ANALYSIS\n")
        # Files
        P_RES_FILE = RES_DIR + "/production_results.xml"
        P_DENS_FILE = RES_DIR + "/production_density.csv"
        NEW_P_RES_FILE = RES_DIR + "/production_clone_length.csv"

        print(" Importing data from " + P_RES_FILE)
        P_LIN_DATA = parseLineageFile(P_RES_FILE)
        if not len(P_LIN_DATA):
            printError("Empty production data: no linages found in " + P_RES_FILE)
            return

        p_last_commit = getLastCommitFromDensityCSV(P_DENS_FILE)
        print("  >> All data imported\n")

        P_LENS = getLenghtsOfDeadLineages(P_LIN_DATA, p_last_commit)
        P_LENS.sort()

        if len(P_LENS) == 0:
            return
        
        MAX = P_LENS[-1]

        with open(NEW_P_RES_FILE,'w+') as p_out:
            for i in range(0,MAX,10):
                ratio = 0
                for l in P_LENS:
                    if l <= i:
                        ratio+=1
                p_out.write(str(i) + "," + str((100*ratio)/float(len(P_LENS))))
                p_out.write("\n")
